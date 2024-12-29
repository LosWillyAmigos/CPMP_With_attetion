from attentional_cpmp.model import create_model
from attentional_cpmp.utils.data_saving import load_data_from_json
from attentional_cpmp.validations import percentage_per_container
from attentional_cpmp.validations import percentage_per_S

from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import AttentionModel

import argparse
import json
import os
import numpy as np
import pandas as pd

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


def train_with_batches(model, data, states, batch_size, epochs):
    """Entrena el modelo utilizando batches por estados."""
    metrics_names = ["loss"] + model.metrics_names[1:]
    history = {metric: [] for metric in metrics_names}

    for epoch in range(epochs):
        epoch_metrics = {metric: [] for metric in metrics_names}

        for batch_start in range(0, max(len(data[state]["States"]) for state in states), batch_size):
            for state in states:
                try:
                    x_train, y_train = get_data(data, state, batch_start, batch_size)
                    metrics = model.train_on_batch(x_train, y_train)
                    for name, value in zip(metrics_names, metrics):
                        epoch_metrics[name].append(value)
                except KeyError:
                    continue

        # Promedio por epoch
        for name in metrics_names:
            history[name].append(sum(epoch_metrics[name]) / len(epoch_metrics[name]))

    return history

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de Keras basado en un archivo JSON de hiperparámetros.")
    parser.add_argument("--json_path", required=True, help="Ruta al archivo JSON con los hiperparámetros.")
    parser.add_argument("--data_path", required=True, help="Ruta al archivo JSON con los datos.")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para guardar modelos, imágenes y el archivo Excel.")
    parser.add_argument("--excel_name", required=True, help="Nombre del archivo Excel de estadísticas (sin ruta).")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch para entrenamiento.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Tamaño del batch para validación.")

    args = parser.parse_args()

    # Cargar el archivo JSON de hiperparámetros
    with open(args.json_path, 'r') as f:
        hyperparams_list = json.load(f)

    # Cargar los datos desde el JSON
    data = load_data_from_json(args.data_path)

    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Entrenar cada modelo con los hiperparámetros dados
    for hyperparams in hyperparams_list:
        model = create_model(**hyperparams)

        states = list(data.keys())
        history = train_with_batches(model, data, states, args.batch_size, epochs=10)

        model.save(f"{args.output_dir + hyperparams["name"]}.h5")
        excel_path = f"{args.output_dir +  args.excel_name}"

        create_statistics(history, excel_path)

        
        for state in states:
            # Guardar la gráfica para cada estado
            plot_path = f"{args.output_dir + hyperparams["name"]}_to_state_{state}_plot.png"

            percentage_per_container(optimizer=GreedyModel(model=model,
                                                        data_adapter=AttentionModel()),
                                    S=int(state),
                                    H=hyperparams["H"],
                                    sample_size=args.sample_size,
                                    save_path=plot_path,
                                    model_name=hyperparams["name"],
                                    excel_path=args.output_dir,
                                    max_steps=int(state) * (hyperparams["H"] - 2) * 2)
        percentage_per_S(optimizer=GreedyModel(model=model,
                                            data_adapter=AttentionModel()),
                        S=[int(state) for state in states],
                        H=hyperparams["H"],
                        N=None,
                        sample_size=args.sample_size,
                        save_path=f"{args.output_dir + hyperparams["name"]}_plot.png",
                        model_name=hyperparams["name"],
                        excel_path=args.output_dir)
if __name__ == "__main__":
    main()
