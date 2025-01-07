from attentional_cpmp.model import create_model
from attentional_cpmp.utils.callbacks.optuna import  BestHyperparameterSaver
from attentional_cpmp.utils.callbacks.optuna import HyperparameterSaver
from attentional_cpmp.validations import validation_per_stack
from attentional_cpmp.utils.train_model import normal_training, batch_training

from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.optimizer import GreedyModel

from keras.callbacks import EarlyStopping
from keras.backend import clear_session

from optuna import Trial
from optuna import Study
from optuna.integration import TFKerasPruningCallback
from optuna.importance import get_param_importances

from typing import Any

import numpy as np
import tensorflow as tf
import json
import os

def objective(trial: Trial,
              max_num_stacks: int,
              max_num_heads: int,
              max_key_dim: int,
              max_value_dim: int,
              max_n_dropout_hide: int,
              max_n_dropout_feed: int,
              max_epsilon: int,
              max_num_neurons_layers_feed: int,
              max_num_neurons_layers_hide: int,
              max_units_neurons_feed: int,
              max_units_neurons_hide: int,
              step:int,
              H: int,
              optimizer: Any | None,
              loss: Any | None,
              metrics: list[Any] | None,
              monitor: str,
              patience: int,
              verbose: int,
              restore_best_weights: bool,
              X_train: Any | np.ndarray,
              Y_train: Any | np.ndarray,
              epochs: int,
              batch_size: int,
              validation_split: float,
              use_saver_callbacks: bool,
              dir_callbacks: str | None = None) -> float:
      
        num_stacks = trial.suggest_int('num_stacks', 1, max_num_stacks, step=step)
        num_heads = trial.suggest_int('num_heads', 1, max_num_heads, step=step)
        key_dim = trial.suggest_int('key_dim', 1, max_key_dim, step=step)

        value_dim = trial.suggest_categorical("value_dim", [None, *range(1, max_value_dim, step)])

        dropout = trial.suggest_float('dropout', 0.0, 0.9)
        rate = trial.suggest_float('rate', 0.0, 0.9)

        activation_hide = trial.suggest_categorical('activation_hide', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        activation_feed = trial.suggest_categorical('activation_feed', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        
        n_dropout_hide = trial.suggest_int('n_dropout_hide', 0, max_n_dropout_hide, step=step)
        n_dropout_feed = trial.suggest_int('n_dropout_feed', 0, max_n_dropout_feed, step=step)

        epsilon = trial.suggest_float('epsilon', 1e-9, max_epsilon, log=True)
        num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 0, max_num_neurons_layers_feed, step=step)
        num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 0, max_num_neurons_layers_hide, step=step)
        list_neurons_feed = [trial.suggest_int(f'list_neurons_feed_{i}', 1, max_units_neurons_feed) for i in range(num_neurons_layers_feed)]
        list_neurons_hide = [trial.suggest_int(f'list_neurons_hide_{i}', 1, max_units_neurons_hide) for i in range(num_neurons_layers_hide)]

        model = create_model(H=H,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            num_heads=num_heads,
                            list_neurons_feed=list_neurons_feed,
                            list_neurons_hide=list_neurons_hide,
                            dropout=dropout,
                            rate=rate,
                            activation_hide=activation_hide,
                            activation_feed=activation_feed,
                            n_dropout_hide=n_dropout_hide,
                            n_dropout_feed=n_dropout_feed,
                            epsilon=epsilon,
                            num_stacks=num_stacks,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)
      
        callbacks = []
        
        pruning_callback = TFKerasPruningCallback(trial, monitor)
        early_stopping_callback = EarlyStopping(
            monitor= monitor,  
            patience=patience,         
            mode='min',          
            verbose=verbose,          
            restore_best_weights=restore_best_weights
        )
        
        callbacks.append(pruning_callback)
        callbacks.append(early_stopping_callback)
        
        if use_saver_callbacks:
            os.makedirs(dir_callbacks, exist_ok=True)
            best_hyp_saver = BestHyperparameterSaver(trial, 
                                                    monitor=monitor, 
                                                    filename=dir_callbacks + "/best_hyp.json",
                                                    metrics=metrics)
            
            all_saver = HyperparameterSaver(trial,
                                            monitor=monitor,
                                            filename=dir_callbacks + "/all_hyp.json",
                                            metrics=metrics)
            callbacks.append(best_hyp_saver)
            callbacks.append(all_saver)
      
        try:
            history = model.fit(x=X_train, 
                                y=Y_train, 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                verbose=verbose, 
                                validation_split=validation_split,
                                callbacks=callbacks)
        except (ValueError, 
                MemoryError, 
                RuntimeError, 
                tf.errors.ResourceExhaustedError) as e:
            print(f"Error en la optimización: {e}")
            raise e
      
        val_loss = history.history[monitor][-1]

        trial.set_user_attr("history", history.history)

        del model

        clear_session()

        return val_loss

def multi_objective(trial: Trial,
              max_num_stacks: int,
              max_num_heads: int,
              max_key_dim: int,
              max_value_dim: int,
              max_n_dropout_hide: int,
              max_n_dropout_feed: int,
              max_epsilon: int,
              max_num_neurons_layers_feed: int,
              max_num_neurons_layers_hide: int,
              max_units_neurons_feed: int,
              max_units_neurons_hide: int,
              H: int,
              step:int = 1,
              optimizer: Any | None = 'Adam',
              loss: Any | None = 'binary_crossentropy',
              metrics: list[Any] | None = ['mae', 'mse'],
              monitor: str = 'val_loss',
              patience: int = 3,
              verbose: int = 1,
              dir_callbacks: str = None,
              restore_best_weights: bool = True,
              data: dict = None,
              epochs: int = 10,
              batch_size: int = 32,
              validation_split: float = 0.2,
              S: list[int] = None,
              use_saver_callbacks: bool = False,
              type_training: str = "normal",
              n_subsets: int = 10,
              max_samples: int | None = None,
              sample_size: int = 1000):
      
        num_stacks = trial.suggest_int('num_stacks', 1, max_num_stacks, step=step)
        num_heads = trial.suggest_int('num_heads', 1, max_num_heads, step=step)
        key_dim = trial.suggest_int('key_dim', 1, max_key_dim, step=step)

        value_dim = trial.suggest_categorical("value_dim", [None, *range(1, max_value_dim, step)])

        dropout = trial.suggest_float('dropout', 0.0, 0.9)
        rate = trial.suggest_float('rate', 0.0, 0.9)

        activation_hide = trial.suggest_categorical('activation_hide', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        activation_feed = trial.suggest_categorical('activation_feed', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        
        n_dropout_hide = trial.suggest_int('n_dropout_hide', 0, max_n_dropout_hide, step=step)
        n_dropout_feed = trial.suggest_int('n_dropout_feed', 0, max_n_dropout_feed, step=step)

        epsilon = trial.suggest_float('epsilon', 1e-9, max_epsilon, log=True)
        num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 0, max_num_neurons_layers_feed, step=step)
        num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 0, max_num_neurons_layers_hide, step=step)
        list_neurons_feed = [trial.suggest_int(f'list_neurons_feed_{i}', 1, max_units_neurons_feed) for i in range(num_neurons_layers_feed)]
        list_neurons_hide = [trial.suggest_int(f'list_neurons_hide_{i}', 1, max_units_neurons_hide) for i in range(num_neurons_layers_hide)]

        model = create_model(H=H, key_dim=key_dim, value_dim=value_dim, 
                            num_heads=num_heads,
                            list_neurons_feed=list_neurons_feed,
                            list_neurons_hide=list_neurons_hide,
                            dropout=dropout, rate=rate,
                            activation_hide=activation_hide,
                            activation_feed=activation_feed,
                            n_dropout_hide=n_dropout_hide,
                            n_dropout_feed=n_dropout_feed,
                            epsilon=epsilon,
                            num_stacks=num_stacks,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)
      
        callbacks = []
        
        early_stopping_callback = EarlyStopping(monitor= monitor, patience=patience,         
            mode='min', verbose=verbose, restore_best_weights=restore_best_weights
        )
        
        callbacks.append(early_stopping_callback)
        
        if use_saver_callbacks:
            os.makedirs(dir_callbacks, exist_ok=True)
            best_hyp_saver = BestHyperparameterSaver(trial,monitor=monitor, 
                                                    filename=dir_callbacks + "/best_hyp.json",
                                                    metrics=metrics)
            all_saver = HyperparameterSaver(trial, monitor=monitor,
                                            filename=dir_callbacks + "/all_hyp.json",
                                            metrics=metrics)
            callbacks.append(best_hyp_saver)
            callbacks.append(all_saver)
      
        try:
            if type_training == "normal":
                state_history = normal_training(model=model,
                                                data=data,
                                                max_samples=max_samples,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                validation_split=validation_split,
                                                callbacks=callbacks,
                                                monitor=monitor,
                                                verbose=verbose)
            elif type_training == "batches":
                state_history = batch_training(model=model,
                                                data=data,
                                                max_samples=max_samples,
                                                n_subsets=n_subsets,
                                                epochs=epochs,
                                                validation_split=validation_split,
                                                batch_size=batch_size,
                                                callbacks=callbacks,
                                                monitor=monitor,
                                                verbose=verbose)
        except (ValueError, 
                MemoryError, 
                RuntimeError, 
                tf.errors.ResourceExhaustedError) as e:
            print(f"Error en la optimización: {e}")
            raise e
        
        percentage = validation_per_stack(optimizer=GreedyModel(model=model, data_adapter=AttentionModel()),
                                          S=S, H=H, N=None, sample_size=sample_size, median_percentage=False)

        mean_val_loss = np.mean(
            [state_history[state][monitor][-1] for state in state_history]
        )
        trial.set_user_attr("history", state_history)
        trial.set_user_attr("percetage", percentage)
        trial.set_user_attr("mean_val_loss", mean_val_loss)

        clear_session()

        return [mean_val_loss, percentage]

def multi_objective_one_set(trial: Trial,
              max_num_stacks: int,
              max_num_heads: int,
              max_key_dim: int,
              max_value_dim: int,
              max_n_dropout_hide: int,
              max_n_dropout_feed: int,
              max_epsilon: int,
              max_num_neurons_layers_feed: int,
              max_num_neurons_layers_hide: int,
              max_units_neurons_feed: int,
              max_units_neurons_hide: int,
              H: int,
              step:int = 1,
              optimizer: Any | None = 'Adam',
              loss: Any | None = 'binary_crossentropy',
              metrics: list[Any] | None = ['mae', 'mse'],
              monitor: str = 'val_loss',
              patience: int = 3,
              verbose: int = 1,
              dir_callbacks: str = None,
              restore_best_weights: bool = True,
              X_train: np.ndarray | tf.Tensor | None = None,
              y_train: np.ndarray | tf.Tensor | None = None,
              epochs: int = 10,
              batch_size: int = 32,
              validation_split: float = 0.2,
              S: int = None,
              use_saver_callbacks: bool = False,
              sample_size: int = 1000):
      
        num_stacks = trial.suggest_int('num_stacks', 1, max_num_stacks, step=step)
        num_heads = trial.suggest_int('num_heads', 1, max_num_heads, step=step)
        key_dim = trial.suggest_int('key_dim', 1, max_key_dim, step=step)

        value_dim = trial.suggest_categorical("value_dim", [None, *range(1, max_value_dim, step)])

        dropout = trial.suggest_float('dropout', 0.0, 0.9)
        rate = trial.suggest_float('rate', 0.0, 0.9)

        activation_hide = trial.suggest_categorical('activation_hide', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        activation_feed = trial.suggest_categorical('activation_feed', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
        
        n_dropout_hide = trial.suggest_int('n_dropout_hide', 0, max_n_dropout_hide, step=step)
        n_dropout_feed = trial.suggest_int('n_dropout_feed', 0, max_n_dropout_feed, step=step)

        epsilon = trial.suggest_float('epsilon', 1e-9, max_epsilon, log=True)
        num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 0, max_num_neurons_layers_feed, step=step)
        num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 0, max_num_neurons_layers_hide, step=step)
        list_neurons_feed = [trial.suggest_int(f'list_neurons_feed_{i}', 1, max_units_neurons_feed) for i in range(num_neurons_layers_feed)]
        list_neurons_hide = [trial.suggest_int(f'list_neurons_hide_{i}', 1, max_units_neurons_hide) for i in range(num_neurons_layers_hide)]

        model = create_model(H=H, key_dim=key_dim, value_dim=value_dim, 
                            num_heads=num_heads,
                            list_neurons_feed=list_neurons_feed,
                            list_neurons_hide=list_neurons_hide,
                            dropout=dropout, rate=rate,
                            activation_hide=activation_hide,
                            activation_feed=activation_feed,
                            n_dropout_hide=n_dropout_hide,
                            n_dropout_feed=n_dropout_feed,
                            epsilon=epsilon,
                            num_stacks=num_stacks,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)
      
        callbacks = []
        
        early_stopping_callback = EarlyStopping(monitor= monitor, patience=patience,         
            mode='min', verbose=verbose, restore_best_weights=restore_best_weights
        )
        
        callbacks.append(early_stopping_callback)
        
        if use_saver_callbacks:
            os.makedirs(dir_callbacks, exist_ok=True)
            best_hyp_saver = BestHyperparameterSaver(trial,monitor=monitor, 
                                                    filename=dir_callbacks + "/best_hyp.json",
                                                    metrics=metrics)
            all_saver = HyperparameterSaver(trial, monitor=monitor,
                                            filename=dir_callbacks + "/all_hyp.json",
                                            metrics=metrics)
            callbacks.append(best_hyp_saver)
            callbacks.append(all_saver)
      
        try:
            history = model.fit(x=X_train, 
                                y=y_train, 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                verbose=verbose, 
                                validation_split=validation_split,
                                callbacks=callbacks)
        except (ValueError, 
                MemoryError, 
                RuntimeError, 
                tf.errors.ResourceExhaustedError) as e:
            print(f"Error en la optimización: {e}")
            raise e
        
        val_monitor = history.history[monitor][-1]

        trial.set_user_attr("history", history.history)
        trial.set_user_attr("val_monitor", val_monitor)

        percentage = validation_per_stack(optimizer=GreedyModel(model=model, data_adapter=AttentionModel()),
                                          S=[S], H=H, N=None, sample_size=sample_size, median_percentage=True)

        trial.set_user_attr("percentage", percentage)

        del model
        clear_session()

        return [val_monitor, percentage]

def insert_trials(path_trials:str = None, study: Study = None) -> None:
    if path_trials is None: 
        raise ValueError("Path to good params is None")
    if study is None: 
        raise ValueError("There isn't a study object")

    params = load_json(path=path_trials)
    
    for params_i in params:
      study.enqueue_trial(params_i)

def load_json(path: str = None) -> dict:
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def show_importances(study: Study) -> None:
    param_importances = get_param_importances(study)

    print("********** Importance of hyperparameters: **********")
    for param, importance in param_importances.items():
        print(f"  {param}: {importance:.8f}")
