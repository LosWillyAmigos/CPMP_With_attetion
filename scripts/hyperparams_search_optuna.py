from attentional_cpmp.utils.hyperparameter_search import HyperparameterStudy
from attentional_cpmp.utils.data_saving.data_json import load_data_from_json
from attentional_cpmp.utils import get_data
import argparse
import random
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
    parser.add_argument('list_cpu', 
                        type=str,
                        help="Lista nucleos a ocupar")
    
    
    # Parametros opcionales
    parser.add_argument('--study_name', '-sn',
                        type=str,
                        help="Nombre del estudio",
                        required=False,
                        default="Study_Model_CPMP")
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
                       default=None)
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
    
    args = parser.parse_args()
    # Limitar el proceso actual
    if args.list_cpu:
        list_core = list(map(int, args.list_cpu.split(",")))
    
    process = psutil.Process(os.getpid())
    process.cpu_affinity(list_core)
    
    
    print("Cargando datos...")
    data_Sx7 = load_data_from_json(args.path_data)
    x, y = get_data(data_Sx7, args.data_dimesion_type)
    del data_Sx7
    print("Datos cargados...")
    print("Empezando creación de estudio...")
    study = HyperparameterStudy(study_name=args.study_name,
                                dir_good_params=args.dir_good_hyp,
                                min_resource=args.min_r,
                                max_resource=args.max_r,
                                reduction_factor=args.red_factor)
    print("Estudio creado")
    print("Configurando estudio...")
    study.set_training_data(X_train=x, Y_train=y,
                            validation_split=args.percentage,
                            epochs=args.epochs, batch_size=args.batch_size)
    
    if args.path_max_config is None:
        study.set_max_config_trial()
    else:
        with open(args.path_max_config, 'r') as file:
            params = json.load(file)
        study.set_max_config_trial(**params)
    
    study.set_config_model(H=args.H, verbose=args.verbose,
                           metrics=['mae', 'mse', 'accuracy'])
    study.set_config_callbacks(dir=args.dir_callback)
    print("Estudio configurado")

    print("Empezando optimización...")

    study.optimize(n_trials=args.n_trials)

    print("Optimización finalizada")