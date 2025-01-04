from optuna import create_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import MedianPruner

from attentional_cpmp.utils.hyperparameter_search import multi_objective, load_json, insert_trials
from attentional_cpmp.utils.data_saving import load_data_from_json
from attentional_cpmp.validations import PercentageSolved

import argparse
import tensorflow as tf
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argumentos para cargar un backend de optuna")
    
    parser.add_argument('--study_name',
                        type=str,
                        required=True,
                        help="Nombre del estudio a cargar")
    parser.add_argument('--storage_name',
                        type=str,
                        required=True,
                        help="Ruta del backend")
    parser.add_argument('--path_data',
                        type=str,
                        required=True,
                        help="Ruta de los datos")
    parser.add_argument('--H',
                        type=int,
                        required=True,
                        help="Dimensión del modelo")
    parser.add_argument('--path_config_model',
                        type=str,
                        required=True,
                        help="Ruta de la configuración del modelo")
    parser.add_argument('--path_config_callbacks',
                        type=str,
                        required=True,
                        help="Ruta de la configuración de los callbacks")
    parser.add_argument('--path_config_max_trials',
                        required=True,
                        type=str,
                        help="Ruta de la configuración de los valores máximos de los trials")
    
    parser.add_argument('--path_good_params',
                        type=str,
                        required=False,
                        help="Ruta de los parametros buenos",
                        default=None)
    parser.add_argument('--type_training',
                        required=False,
                        type=str,
                        help="Tipo de entrenamiento para la optimización",
                        default="batches")
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
    parser.add_argument('--n_subsets',
                        type=int,
                        help="Cantidad de subconjuntos",
                        required=False,
                        default=5)
    parser.add_argument('--max_samples',
                        type=int,
                        help="Cantidad máxima de muestras",
                        required=False,
                        default=None)
    parser.add_argument('--sample_size',
                        type=int,
                        help="Cantidad máxima de muestras",
                        required=False,
                        default=1000)
    
    args = parser.parse_args()
    
    study = create_study(study_name=args.study_name, 
                         pruner=MedianPruner(),
                         storage=JournalStorage(JournalFileBackend(args.storage_name)),
                         directions=["minimize", "maximize"],
                         load_if_exists=True)
    if args.path_good_params is not None:
        insert_trials(path_trials=args.path_good_params, study=study)
    
    data = load_data_from_json(args.path_data)

    S = [int(state) for state in data.keys()]
    
    config_model = load_json(args.path_config_model)
    config_model["metrics"].append(PercentageSolved())
    config_max_trials = load_json(args.path_config_max_trials)
    config_callbacks = load_json(args.path_config_callbacks)

    os.makedirs(config_callbacks['dir_callbacks'], exist_ok=True)
    
    study.optimize(lambda trial: multi_objective(trial,
                                        H=args.H,
                                        data=data,
                                        type_training=args.type_training,
                                        n_subsets=args.n_subsets,
                                        max_samples=args.max_samples,
                                        sample_size=args.sample_size,
                                        S=S,
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