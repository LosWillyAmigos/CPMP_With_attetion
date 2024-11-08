from attentional_cpmp.model import create_model
from attentional_cpmp.utils.callbacks import  BestHyperparameterSaver
from attentional_cpmp.utils import create_directory

from optuna.integration import TFKerasPruningCallback
from keras.callbacks import EarlyStopping

from optuna.visualization import plot_param_importances
from optuna.importance import get_param_importances
from optuna.pruners import HyperbandPruner
from optuna import create_study
from keras.backend import clear_session

import numpy as np
import os
import json

class HyperparameterStudy:
  def __init__(self,
                study_name: str = "Study_Model_CPMP", 
                direction: str = 'minimize', 
                min_resource: int = 1, 
                max_resource: int = 100,
                reduction_factor: int = 3,
                dir_good_params: str = None) -> None:    

    # Param optimization
    self.__pruner = HyperbandPruner(min_resource=min_resource,
                                    max_resource=max_resource, 
                                    reduction_factor=reduction_factor)

    self.__study = create_study(study_name=study_name, 
                                direction=direction, 
                                pruner=self.__pruner)

    self.inser_manual_trials(dir_good_params)

  def objective(self, trial):
      clear_session()
      # Hiperparametros variables
      num_stacks = trial.suggest_int('num_stacks', 1, self.__max_num_stacks)
      num_heads = trial.suggest_int('num_heads', 1, self.__max_num_heads)
      key_dim = trial.suggest_int('key_dim', 1, self.__max_key_dim)

      value_dim = trial.suggest_int('value_dim', 0, self.__max_value_dim)
      if value_dim == 0:
          value_dim = None

      dropout = trial.suggest_float('dropout', 0.0, 1.0, step=0.1)
      rate = trial.suggest_float('param', 0.0, 1.0, step=0.1)

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
      
      ## Callbacks
      pruning_callback = TFKerasPruningCallback(trial, self.__metrics_monitor_callback)
      
      early_stopping_callback = EarlyStopping(
          monitor= self.__metrics_monitor_callback,  
          patience=self.__patience,         
          mode='min',          
          verbose=self.__verbose_callback,          
          restore_best_weights=self.__restore_best_weights
      )
      
      best_hyp_saver = BestHyperparameterSaver(trial, 
                                               monitor=self.__metrics_monitor_callback, 
                                               filename=self.__dir + "best_hyperparameter.json")
      
      history = model.fit(x=self.__X_train, 
                          y=self.__Y_train, 
                          epochs=self.__epochs, 
                          batch_size=self.__batch_size, 
                          verbose=self.__verbose, 
                          validation_data=(self.__X_val, self.__Y_val),
                          callbacks=[pruning_callback, 
                                     early_stopping_callback,
                                     best_hyp_saver])
      
      clear_session()
      val_metric = history.history[self.__metrics_monitor_callback][-1]

      return val_metric
  
  def set_config_model(self,
                      H: int,
                      optimizer: str | None = 'Adam',
                      loss: str = 'binary_crossentropy',
                      metrics: list[str] = ['mae', 'mse'],
                      verbose:int = 0):
    self.__H = H
    self.__optimizer = optimizer
    self.__loss = loss
    self.__metrics = metrics
    self.__verbose = verbose

  def set_max_config_trial(self,
                           max_num_stacks: int = 15,
                           max_num_heads: int = 15,
                           max_key_dim: int = 128,
                           max_value_dim: int = 128,
                           max_epsilon: float = 1e-3,
                           max_num_neurons_layers_feed: int = 100,
                           max_num_neurons_layers_hide: int = 100,
                           max_units_neurons_hide: int = 100,
                           max_units_neurons_feed: int = 100,
                           max_n_dropout_hide: int = 5,
                           max_n_dropout_feed: int = 5):
    
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
  
  def set_training_data(self,
                        X_train: np.ndarray = None, 
                        Y_train: np.ndarray = None, 
                        X_val: np.ndarray = None, 
                        Y_val: np.ndarray = None, 
                        epochs: int = 20, 
                        batch_size: int = 32) -> None:
    self.__X_train = X_train
    self.__Y_train = Y_train
    self.__X_val = X_val
    self.__Y_val = Y_val
    self.__epochs = epochs
    self.__batch_size = batch_size

  def set_config_callbacks(self,
                           dir: str = None,
                           patience: int = 1,
                           verbose: int = 1,
                           restore_best_weights: bool = True,
                           metrics_monitor_callback: str = 'val_loss'):
    if dir is None:
       create_directory("hyperparameter_test/")
       self.__dir = "hyperparameter_test/"
    else: self.__dir = dir

    if not os.path.exists(self.__dir + "best_hyperparameter.json"):
      with open(self.__dir + "best_hyperparameter.json", 'w', encoding='utf-8') as file:
        pass
    # Callback config
    self.__patience = patience
    self.__verbose_callback = verbose
    self.__restore_best_weights = restore_best_weights
    self.__metrics_monitor_callback = metrics_monitor_callback

  def inser_manual_trials(self, dir_good_params:str):
    if dir_good_params is None: return

    with open(dir_good_params, 'r') as file:
      params = json.load(file)
    
    self.__study.enqueue_trial(params)

  def optimize(self, n_trials, show_progress_bar=True):
    self.__study.optimize(func=self.objective, 
                          n_trials=n_trials, 
                          show_progress_bar=show_progress_bar)

  def importance(self):
    param_importances = get_param_importances(self.__study)

    print("********** Importance of hyperparameters: **********")
    for param, importance in param_importances.items():
        print(f"  {param}: {importance:.8f}")
  
  def display(self):
    plot_param_importances(self.__study)
  
  def save(self, filename: str = None):
    if filename is None: filename = self.__study_name
    with open(filename + '.hyp', 'w') as custom_file:
      best_params = self.__study.best_params
      for clave, valor in best_params.items():
          custom_file.write(f"{clave}: {valor}\n")