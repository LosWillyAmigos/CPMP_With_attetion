from attentional_cpmp.utils.data_saving.data_json import load_data_from_json
from attentional_cpmp.utils.hyperparameter_search import build_model
from attentional_cpmp.utils import get_data
from keras.metrics import Precision
from keras.callbacks import EarlyStopping

import argparse
import json
import psutil
import os

import keras_tuner as kt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ejemplo de argparse con parámetros obligatorios y opcionales")
    # Parametros obligatorios
    parser.add_argument('dir_data',
                        type=str, 
                        help="Ruta de los datos")
    parser.add_argument('dim_data',
                         type=str, 
                         help="Dimensión de los datos a recuperar.")
    parser.add_argument('percentage',
                        type=float, 
                        help="Porcentaje de la data de validación y entramiento.")
    parser.add_argument('H',
                        type=int,
                        help="Configura la entrada del modelo.")
    parser.add_argument('dir_max_config',
                        type=str,
                        help="Directorio de archivo para configuración del estudio.",
                        default=None)
    parser.add_argument('tuner',
                        type=str,
                        help="tuner a usar en las pruebas.",
                        default="random")
    parser.add_argument('list_cpu', 
                        type=str,
                        help="Lista nucleos a ocupar")

    parser.add_argument('--dir_tuner',
                       type=str,
                       help="Dirección requerida por los tuner",
                       required=False,
                       default=None)
    parser.add_argument('--monitor',
                       type=str,
                       help="Metrica de perdida a monitorear",
                       required=False,
                       default='val_loss')
    parser.add_argument('--max_trials',
                       type=int,
                       help="Máximo de pruebas",
                       required=False,
                       default=10)
    parser.add_argument('--name_model',
                       type=str,
                       help="Nombre del modelo",
                       required=False,
                       default='best_model')
    parser.add_argument('--batch_size',
                       type=int,
                       help="Máximo del batch",
                       required=False,
                       default=32)
    parser.add_argument('--epochs',
                       type=int,
                       help="Cantidad de épocas",
                       required=False,
                       default=10)
    parser.add_argument('--executions_per_trial',
                       type=int,
                       help="Cantidad de entrenamientos por prueba",
                       required=False,
                       default=2)
    
    args = parser.parse_args()

    if args.list_cpu:
        list_core = list(map(int, args.list_cpu.split(",")))
    
    process = psutil.Process(os.getpid())
    process.cpu_affinity(list_core)

    print("Cargando datos...")

    data_Sx7 = load_data_from_json(args.dir_data)
    (x_train, y_train) = get_data(data_Sx7, args.dim_data)
    del data_Sx7

    print("Datos cargados...")

    with open(args.dir_max_config, 'r') as file:
        params = json.load(file)

    params["H"] = args.H
    params["loss"] = "binary_crossentropy"
    params["metrics"] = ["mae", "mse", "accuracy"]

    if args.tuner == "hyperband":
        tuner = kt.Hyperband(lambda hp: build_model(hp, **params),
                          objective=args.monitor,
                          max_epochs=args.max_trials,
                          factor=3,
                          hyperband_iterations=1,
                          seed=None,
                          hyperparameters=None,
                          tune_new_entries=True,
                          allow_new_entries=True,
                          max_retries_per_trial=0,
                          max_consecutive_failed_trials=3,
                          directory=args.dir_tuner,
                          executions_per_trial=args.executions_per_trial)
    elif args.tuner == "bayesian":
        tuner = kt.BayesianOptimization(lambda hp: build_model(hp, **params),
                                    objective="val_loss",
                                    max_trials=args.max_trials,
                                    num_initial_points=None,
                                    alpha=0.0001,
                                    beta=2.6,
                                    seed=None,
                                    hyperparameters=None,
                                    tune_new_entries=True,
                                    allow_new_entries=True,
                                    max_retries_per_trial=0,
                                    max_consecutive_failed_trials=3,
                                    directory=args.dir_tuner,
                                    executions_per_trial=args.executions_per_trial)
    elif args.tuner == "random":
        tuner = kt.RandomSearch(lambda hp: build_model(hp, **params),
                                objective="val_loss",
                                max_trials=args.max_trials,
                                seed=None,
                                hyperparameters=None,
                                tune_new_entries=True,
                                allow_new_entries=True,
                                max_retries_per_trial=0,
                                max_consecutive_failed_trials=3,
                                directory=args.dir_tuner,
                                executions_per_trial=args.executions_per_trial)
        
    tuner.results_summary()
    
    tuner.search(x_train, y_train, 
                 epochs=args.epochs, 
                 validation_split=args.percentage,
                 batch_size=args.batch_size,
                 callbacks = [EarlyStopping(monitor='val_loss', 
                                            patience=3,
                                            mode="min",
                                            verbose=1)],
                 verbose=1)
    

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps_dict = best_hps.values

    with open('best_hyperparameters_keras.json', 'w') as json_file:
        json.dump(best_hps_dict, json_file, indent=4)

    print("Hiperparámetros guardados en 'best_hyperparameters_keras.json'")

    best_model = tuner.hypermodel.build(best_hps)

    best_model.fit(x_train, 
                   y_train, 
                   epochs=args.epochs, 
                   validation_split=args.percentage,
                   batch_size=args.batch_size)
    best_model.save(args.name_model + '.h5')
