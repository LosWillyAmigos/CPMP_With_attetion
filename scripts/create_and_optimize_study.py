from optuna import create_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import MedianPruner

from attentional_cpmp.utils.hyperparameter_search import objective, load_json, insert_trials
from attentional_cpmp.utils.data_saving import load_data_from_json
from attentional_cpmp.utils import get_data

import argparse
import tensorflow as tf

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argumentos para cargar un backend de optuna")
    
    parser.add_argument('study_name',
                        type=str,
                        help="Nombre del estudio a cargar")
    parser.add_argument('storage_name',
                        type=str,
                        help="Ruta del backend")
    parser.add_argument('path_good_params',
                        type=str,
                        help="Ruta de los parametros buenos")
    parser.add_argument('path_data',
                        type=str,
                        help="Ruta de los datos")
    parser.add_argument('dim_data',
                        type=str,
                        help="Dimensión de los datos a recuperar")
    parser.add_argument('H',
                        type=int,
                        help="Dimensión del modelo")
    parser.add_argument('validation_split',
                        type=float,
                        help="porcentaje de validación")
    parser.add_argument('path_config_model',
                        type=str,
                        help="Ruta de la configuración del modelo")
    parser.add_argument('path_config_callbacks',
                        type=str,
                        help="Ruta de la configuración de los callbacks")
    parser.add_argument('path_config_max_trials',
                        type=str,
                        help="Ruta de la configuración de los valores máximos de los trials")
    
    parser.add_argument('--n_trials',
                        type=int,
                        help="Cantidad de pruebas",
                        required=False,
                        default=1)
    parser.add_argument('--n_jobs',
                        type=int,
                        help="Cantidad de trabajos en paralelo",
                        required=False,
                        default=1)
    
    
    args = parser.parse_args()
    
    study = create_study(study_name=args.study_name, 
                         pruner=MedianPruner(),
                         storage=JournalStorage(JournalFileBackend(args.storage_name)),
                         direction="minimize",
                         load_if_exists=True)
    
    insert_trials(path_trials=args.path_good_params, study=study)
    
    data_Sx7 = load_data_from_json(args.path_data)
    X_train, Y_train = get_data(data_Sx7, args.dim_data)
    
    config_model = load_json(args.path_config_model)
    config_max_trials = load_json(args.path_config_max_trials)
    config_callbacks = load_json(args.path_config_callbacks)
    
    study.optimize(lambda trial: objective(trial,
                                        H=args.H,
                                        X_train=X_train,
                                        Y_train=Y_train,
                                        **config_callbacks,
                                        **config_max_trials,
                                        **config_model),
                   n_trials=args.n_trials, 
                   show_progress_bar=True,
                   gc_after_trial=True,
                   n_jobs=args.n_jobs,
                   catch=[ValueError,  
                          MemoryError, 
                          RuntimeError, 
                          tf.errors.ResourceExhaustedError])