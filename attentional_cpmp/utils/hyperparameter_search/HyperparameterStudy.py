from attentional_cpmp.model import create_model
from attentional_cpmp.utils import create_directory
from attentional_cpmp.utils.callbacks.optuna import  BestHyperparameterSaver
from attentional_cpmp.utils.callbacks.optuna import HyperparameterSaver

import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.backend import clear_session

from optuna import Trial
from optuna import create_study
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.integration import TFKerasPruningCallback
from optuna.importance import get_param_importances


from typing import Any

import numpy as np
import os
import json

class HyperparameterStudy:
  def __init__(self,
                study_name: str = "Study_Model_CPMP", 
                direction: str = 'minimize', 
                pruner: BasePruner = None,
                sampler: BaseSampler = None,
                storage: BaseStorage = None,
                dir_good_params: str = None,
                load_if_exists: bool = True) -> None:    

    self.__study = create_study(study_name=study_name, 
                                direction=direction, 
                                pruner=pruner,
                                storage=storage,
                                sampler=sampler,
                                load_if_exists=load_if_exists)

    self.inser_manual_trials(dir_good_params)

  def objective(self, trial: Trial):
      num_stacks = trial.suggest_int('num_stacks', 1, self.__max_num_stacks)
      num_heads = trial.suggest_int('num_heads', 1, self.__max_num_heads)
      key_dim = trial.suggest_int('key_dim', 1, self.__max_key_dim)

      value_dim = trial.suggest_categorical("value_dim", [None, *range(1, self.__max_value_dim)])

      dropout = trial.suggest_float('dropout', 0.0, 0.9)
      rate = trial.suggest_float('rate', 0.0, 0.9)

      activation_hide = trial.suggest_categorical('activation_hide', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
      activation_feed = trial.suggest_categorical('activation_feed', ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
      
      n_dropout_hide = trial.suggest_int('n_dropout_hide', 0, self.__max_n_dropout_hide)
      n_dropout_feed = trial.suggest_int('n_dropout_feed', 0, self.__max_n_dropout_feed)

      epsilon = trial.suggest_float('epsilon', 1e-9, self.__max_epsilon, log=True)
      num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 0, self.__max_num_neurons_layers_feed)
      num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 0, self.__max_num_neurons_layers_hide)
      list_neurons_feed = [trial.suggest_int(f'list_neurons_feed_{i}', 1, self.__max_units_neurons_feed) for i in range(num_neurons_layers_feed)]
      list_neurons_hide = [trial.suggest_int(f'list_neurons_hide_{i}', 1, self.__max_units_neurons_hide) for i in range(num_neurons_layers_hide)]

      model = create_model(H=self.__H,
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
                           optimizer=self.__optimizer,
                           loss=self.__loss,
                           metrics=self.__metrics)
      
      pruning_callback = TFKerasPruningCallback(trial, self.__monitor)
      
      early_stopping_callback = EarlyStopping(
          monitor= self.__monitor,  
          patience=self.__patience,         
          mode='min',          
          verbose=self.__verbose_callback,          
          restore_best_weights=self.__restore_best_weights
      )
      
      best_hyp_saver = BestHyperparameterSaver(trial, 
                                               monitor=self.__monitor, 
                                               filename=self.__dir + self.__filename_best_hyp,
                                               metrics=self.__metrics)
      
      all_saver = HyperparameterSaver(trial,
                                      monitor=self.__monitor,
                                      filename=self.__dir + self.__filename_all_hyp,
                                      metrics=self.__metrics)
      
      
      try:
          history = model.fit(x=self.__X_train, 
                              y=self.__Y_train, 
                              epochs=self.__epochs, 
                              batch_size=self.__batch_size, 
                              verbose=self.__verbose, 
                              validation_split=self.__validation_split,
                              callbacks=[pruning_callback, 
                                        early_stopping_callback,
                                        all_saver,
                                        best_hyp_saver])
      except (ValueError, 
              MemoryError, 
              RuntimeError, 
              tf.errors.ResourceExhaustedError) as e:
          print(f"Error en la optimización: {e}")
          raise e
      
      val_loss = history.history[self.__monitor][-1]

      trial.set_user_attr("history", history.history)

      del model

      clear_session()

      return val_loss
  
  def set_config_model(self,
                      H: int,
                      X_train: Any | np.ndarray = None, 
                      Y_train: Any | np.ndarray = None, 
                      validation_split: float = 0.2,
                      epochs: int = 10, 
                      batch_size: int = 32,
                      optimizer: str | None = 'Adam',
                      loss: str = 'binary_crossentropy',
                      metrics: list[Any] = None,
                      verbose:int = 0) -> None:
         
    if X_train is None or Y_train is None:
      raise ValueError("Data to train model is None")
    
    self.__H = H
    self.__optimizer = optimizer
    self.__loss = loss
    self.__metrics = metrics
    self.__verbose = verbose
                        
    self.__X_train = X_train
    self.__Y_train = Y_train
    self.__validation_split = validation_split
    self.__epochs = epochs
    self.__batch_size = batch_size

  def set_max_config_trial(self,
                           max_num_stacks: int = 10,
                           max_num_heads: int = 10,
                           max_key_dim: int = 128,
                           max_value_dim: int = 128,
                           max_epsilon: float = 1e-3,
                           max_num_neurons_layers_feed: int = 50,
                           max_num_neurons_layers_hide: int = 50,
                           max_units_neurons_hide: int = 128,
                           max_units_neurons_feed: int = 128,
                           max_n_dropout_hide: int = 10,
                           max_n_dropout_feed: int = 10):
    
    self.__max_num_stacks = max_num_stacks
    self.__max_num_heads = max_num_heads
    self.__max_key_dim = max_key_dim
    self.__max_value_dim = max_value_dim
    self.__max_epsilon = max_epsilon
    self.__max_num_neurons_layers_feed = max_num_neurons_layers_feed
    self.__max_num_neurons_layers_hide = max_num_neurons_layers_hide
    self.__max_units_neurons_hide = max_units_neurons_hide
    self.__max_units_neurons_feed = max_units_neurons_feed
    self.__max_n_dropout_hide = max_n_dropout_hide
    self.__max_n_dropout_feed = max_n_dropout_feed

  def set_config_callbacks(self,
                           patience: int = 1,
                           verbose: int = 1,
                           restore_best_weights: bool = True,
                           monitor: str = 'val_loss',
                           dir: str = "./hyperparameter_test/",
                           filename_best_hyp: str = "best_hyperparameter",
                           filename_all_hyp: str = "all_hyperparameters"):
    self.__dir = dir
    self.__filename_best_hyp = filename_best_hyp + ".json"
    self.__filename_all_hyp = filename_all_hyp + ".json"

    if not os.path.exists(self.__dir):
       create_directory(self.__dir)

    if not os.path.exists(self.__dir + self.__filename_best_hyp):
      with open(self.__dir + self.__filename_best_hyp, 'w', encoding='utf-8') as file:
        pass

    if not os.path.exists(self.__dir + self.__filename_all_hyp):
      with open(self.__dir + self.__filename_all_hyp, 'w', encoding='utf-8') as file:
        pass
    # Callback config
    self.__patience = patience
    self.__verbose_callback = verbose
    self.__restore_best_weights = restore_best_weights
    self.__monitor = monitor

  def inser_manual_trials(self, dir_good_params:str):
    if dir_good_params is None: return

    with open(dir_good_params, 'r') as file:
      params = json.load(file)
    
    for params_i in params:
      self.__study.enqueue_trial(params_i)

  def optimize(self, n_trials:int, n_jobs:int, show_progress_bar:bool = True):
    self.__study.optimize(func=self.objective, 
                          n_trials=n_trials, 
                          show_progress_bar=show_progress_bar,
                          gc_after_trial=True,
                          n_jobs=n_jobs,
                          catch=[ValueError,   
                                MemoryError, 
                                RuntimeError, 
                                tf.errors.ResourceExhaustedError])

  def importance(self):
    param_importances = get_param_importances(self.__study)

    print("********** Importance of hyperparameters: **********")
    for param, importance in param_importances.items():
        print(f"  {param}: {importance:.8f}")
