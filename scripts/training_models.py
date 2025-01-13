from attentional_cpmp.model import create_model
from attentional_cpmp.utils.data_saving import load_data_from_json
from attentional_cpmp.validations import validation_optimizer_per_container
from attentional_cpmp.validations import validation_optimizer_per_stack
from attentional_cpmp.utils import get_config_model

from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import AttentionModel

from keras.backend import clear_session
from keras.callbacks import EarlyStopping

import argparse
import json
import os
import numpy as np
import pandas as pd
import gc


def get_data(data, state, batch_start, batch_size):
    """Obtiene un batch de datos desde un estado específico."""
    subset = data.get(state)
    if not subset:
        raise KeyError(f"No existe el estado {state} en el diccionario.")

    states = subset["States"]
    labels = subset["Labels"]

    batch_end = min(batch_start + batch_size, len(states))
    return np.stack(states[batch_start:batch_end]), np.stack(labels[batch_start:batch_end])


def create_statistics(train_history, output_path):
    """Guarda estadísticas del entrenamiento en un archivo Excel."""
    stats = {"Epoch": list(range(1, len(next(iter(train_history.values()))) + 1))}
    stats.update(train_history)
    stats_df = pd.DataFrame(stats)
    stats_df.to_excel(output_path, sheet_name="Training History", index=False)

def create_statistics_2(history, excel_path, sheet_name):
    stats = {"Epoch": list(range(1, len(next(iter(history.values()))) + 1))}
    stats.update(history)
    stats_df = pd.DataFrame(stats)

    # Guardar en Excel añadiendo páginas
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a' if os.path.exists(excel_path) else 'w') as writer:
        stats_df.to_excel(writer, sheet_name=sheet_name, index=False)


def train_with_batches(model, data, states, batch_size, epochs, max_samples=None, metrics = None):
    """Entrena el modelo utilizando batches por estados, con la opción de limitar el total de datos."""
    # Limitar datos si max_samples está definido
    if max_samples is not None:
        for state in states:
            data[state]["States"] = data[state]["States"][:max_samples]
            data[state]["Labels"] = data[state]["Labels"][:max_samples]

    metrics_names = ["loss"] + metrics
    history = {metric: [] for metric in metrics_names}

    for epoch in range(epochs):
        epoch_metrics = {metric: [] for metric in metrics_names}
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_start in range(0, (max_samples if max_samples is not None else max(len(data[state]["States"]) for state in states)), batch_size):
            for state in states:
                try:
                    x_train, y_train = get_data(data, state, batch_start, batch_size)
                    metrics = model.train_on_batch(x_train, y_train, return_dict=True)
                    for name in metrics_names:
                        epoch_metrics[name].append(metrics[name])
                except KeyError:
                    continue
                print(f"Batch {batch_start // batch_size + 1}/{(max_samples if max_samples is not None else len(data[state]["States"])) // batch_size} - State {state} - Loss: {metrics['loss']:.6f}")

        # Promedio por epoch
        for name in metrics_names:
            history[name].append(sum(epoch_metrics[name]) / len(epoch_metrics[name]))

    return history


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de Keras basado en un archivo JSON de hiperparámetros.")
    parser.add_argument("--json_path", required=True, help="Ruta al archivo JSON con los hiperparámetros.")
    parser.add_argument("--data_path", required=True, help="Ruta al archivo JSON con los datos.")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para guardar modelos, imágenes y el archivo Excel.")
    parser.add_argument("--excel_name", required=False, default="statics.xlsx", help="Nombre del archivo Excel de estadísticas (sin ruta).")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch para entrenamiento.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Tamaño del batch para validación.")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento.")
    parser.add_argument("--max_samples", type=int, default=None, help="Número máximo de datos a usar por estado (None para usar todos los datos).")
    parser.add_argument("--training_with_batches", type=int, default=1, help="Entrenar con batches")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Porcentaje de datos para validación.")

    args = parser.parse_args()

    # Cargar el archivo JSON de hiperparámetros
    with open(args.json_path, 'r') as f:
        hyperparams_list = json.load(f)

    # Cargar los datos desde el JSON
    print("Cargando datos...")
    data = load_data_from_json(args.data_path)

    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Entrenar cada modelo con los hiperparámetros dados
    for hyperparams in hyperparams_list:
        os.makedirs(args.output_dir + hyperparams['name'] + '/', exist_ok=True)
        print(f"Entrenando modelo {hyperparams['name']}...")
        model = create_model(**get_config_model(hyperparams))

        states = list(data.keys())
        excel_path = f"{args.output_dir + hyperparams['name'] + '/' +  args.excel_name}"
        if args.training_with_batches == 1:
            print("Entrenando con batches...")
            history = train_with_batches(model, 
                                        data, 
                                        states, 
                                        args.batch_size, 
                                        epochs=args.epochs, 
                                        max_samples=args.max_samples, 
                                        metrics = hyperparams['metrics'])
            create_statistics(history, excel_path)
        elif args.training_with_batches == 0:
            if args.max_samples is not None:
                for state in states:
                    data[state]["States"] = data[state]["States"][:args.max_samples]
                    data[state]["Labels"] = data[state]["Labels"][:args.max_samples]
            for stack in data:
                history = model.fit(np.stack(data[stack]["States"]), np.stack(data[stack]["Labels"]), 
                                    batch_size=args.batch_size, epochs=args.epochs, validation_split=args.validation_split,
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
                data.get(stack)
                create_statistics_2(history.history, args.output_dir + hyperparams['name'] + '_' + args.excel_name, str(stack))

        print("Guardando modelo y estadísticas...")
        model.save(f"{args.output_dir + hyperparams['name'] + '/' + hyperparams["name"]}.keras")
        model.save(f"{args.output_dir + hyperparams['name'] + '/' + hyperparams["name"]}.h5")

        print("Generando gráficas...")
        
        validation_optimizer_per_container(optimizers=[GreedyModel(model=model, data_adapter=AttentionModel()), GreedyV2()],
                                           optimizers_name=["GreedyModel", "Greedy2"],
                                           S=states,
                                           H=hyperparams["H"],
                                           sample_size=args.sample_size,
                                           output_dir=args.output_dir + hyperparams['name'] + '/',
                                           calculate_only_solved=True,
                                           hyperparameter_name=hyperparams["name"])
        validation_optimizer_per_stack(optimizers=[GreedyModel(model=model, data_adapter=AttentionModel()), GreedyV2()],
                                       optimizers_name=["GreedyModel", "Greedy2"],
                                       S=states,
                                       H=hyperparams["H"],
                                       N=1,
                                       sample_size=args.sample_size,
                                       output_dir=args.output_dir + hyperparams['name'] + '/',
                                       calculate_only_solved=True,
                                       hyperparameter_name=hyperparams["name"])
        clear_session()
        gc.collect()
        
if __name__ == "__main__":
    main()
