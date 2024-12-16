from attentional_cpmp.utils.hyperparameter_search import HyperparameterStudy
from attentional_cpmp.utils.data_saving.data_json import load_data_from_json
from attentional_cpmp.utils import get_data

from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from datetime import datetime

import argparse
import json
import psutil
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ejemplo de argparse con parámetros obligatorios y opcionales")

    # Parametros obligatorios
    parser.add_argument('path_data',
                        type=str, 
                        help="Ruta de los ratos")
    parser.add_argument('path_max_config',
                        type=str,
                        help="Directorio de archivo para configuración del estudio.",
                        default=None)
    parser.add_argument('data_dimesion_type',
                         type=str, 
                         help="Dimensión de los datos a recuperar.")
    parser.add_argument('percentage',
                        type=float, 
                        help="Porcentaje de la data de validación y entramiento.")
    parser.add_argument('H',
                        type=int,
                        help="Configura la entrada del modelo.")
    
    
    # Parametros opcionales
    parser.add_argument('--list_cpu', 
                        type=str,
                        help="Lista nucleos a ocupar",
                        required=False,
                        default=None)
    parser.add_argument('--study_name', '-sn',
                        type=str,
                        help="Nombre del estudio",
                        required=None,
                        default=None)
    parser.add_argument('--dir_good_hyp', '-gh',
                        type=str,
                        help="Directorio de hiperparametros sugeridos.",
                        required=False,
                        default=None)
    parser.add_argument('--min_r', '-mir',
                        type=int,
                        help="Controla el punto de partida de recursos de cada ensayo.",
                        required=False,
                        default=1)
    parser.add_argument('--max_r', '-mar',
                        type=int,
                        help="Establece el máximo de recursos que un ensayo puede usar.",
                        required=False,
                        default=10)
    parser.add_argument('--red_factor', '-rf',
                        type=int,
                        help="Determina cuántos ensayos pasan a la siguiente ronda.",
                        required=False,
                        default=5)
    parser.add_argument('--verbose', '-v',
                        type=int,
                        help="Muestra el progreso de entrenamiento del modelo.",
                        required=False,
                        default=0)
    parser.add_argument('--n_trials', '-nt',
                        type=int,
                        default=10,
                        required=False,
                        help="Número de pruebas a realizar.")
    parser.add_argument('--dir_callback', '-dc',
                       type=str,
                       help="Dirección requerida por los callbacks",
                       required=False,
                       default="./hyperparameter_test/")
    parser.add_argument('--storage_name',
                       type=str,
                       help="Nombre del storage que guarda los estudios",
                       required=False,
                       default="Study_CPMP_log.json")
    parser.add_argument('--epochs',
                       type=int,
                       help="Cantidad de epocas de entrenamiento",
                       required=False,
                       default=10)
    parser.add_argument('--batch_size',
                       type=int,
                       help="Cantidad de batch de entrenamiento",
                       required=False,
                       default=32)
    parser.add_argument('--n_jobs',
                       type=int,
                       help="Cantidad nucleos a usar en paralelo",
                       required=False,
                       default=1)
    parser.add_argument('--load_if_exists',
                       type=bool,
                       help="Cargar backend si existe",
                       required=False,
                       default=False)
    
    args = parser.parse_args()
    # Limitar el proceso actual
    if args.list_cpu is not None:
        list_core = list(map(int, args.list_cpu.split(",")))
    
        process = psutil.Process(os.getpid())
        process.cpu_affinity(list_core)
    
    
    print("Loading data...")
    data_Sx7 = load_data_from_json(args.path_data)
    x, y = get_data(data_Sx7, args.data_dimesion_type)
    del data_Sx7
    print("Data loaded!")

    print("Starting study creation...")
    if args.study_name is None:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d_%H:%M:%S")
        study_name = f"Study_CPMP_{date}"
    else:
        study_name = args.study_name
    
    if args.storage_name is None:
        storage_name = "Study_CPMP_log.json"
    else:
        storage_name = args.storage_name

    study = HyperparameterStudy(study_name=study_name,
                                dir_good_params=args.dir_good_hyp,
                                pruner=MedianPruner(),
                                storage=JournalStorage(JournalFileBackend(storage_name)),
                                load_if_exists=args.load_if_exists)
    print("Study created")
    print("Setting up study...")
    
    if args.path_max_config is None:
        study.set_max_config_trial()
    else:
        with open(args.path_max_config, 'r') as file:
            params = json.load(file)
        study.set_max_config_trial(**params)
    
    study.set_config_model(H=args.H, verbose=args.verbose,
                           metrics=['mae', 'mse', 'accuracy'],
                           X_train=x, Y_train=y,
                           validation_split=args.percentage,
                           epochs=args.epochs, batch_size=args.batch_size)
    study.set_config_callbacks(dir=args.dir_callback, patience=2)
    print("Study set up")

    print("Starting optimization...")

    study.optimize(n_trials=args.n_trials,
                   n_jobs=args.n_jobs)

    print("Optimization completed")