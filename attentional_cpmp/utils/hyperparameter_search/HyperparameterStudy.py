from attentional_cpmp.model import create_model
from attentional_cpmp.utils.callbacks import  BestHyperparameterSaver
from attentional_cpmp.utils import create_directory

from optuna.integration import TFKerasPruningCallback
from optuna.importance import get_param_importances
from optuna.pruners import HyperbandPruner
from optuna import create_study
from keras.backend import clear_session
from keras.callbacks import EarlyStopping

import numpy as np
import os

class HyperparameterStudy:
  def __init__(self, 
                H: int, 
                optimizer: str = 'Adam', 
                X_train: np.ndarray = None, 
                Y_train: np.ndarray = None, 
                X_val: np.ndarray = None, 
                Y_val: np.ndarray = None, 
                epochs: int = 20, 
                batch_size: int = 32,
                study_name: str = "Study_Model_CPMP", 
                direction: str = 'minimize', 
                min_resource: int = 1, 
                max_resource: int = 100,
                reduction_factor: int = 3,
                number_of_iterations: int = 10,
                dir: str = None,
                patience: int = 3,
                verbose: int = 1,
                restore_best_weights: bool = True) -> None:
    # Exceptions
    if X_train is None or Y_train is None or X_val is None or Y_val is None:
       raise ValueError("There is a problem with data...")
    
    # Create a directory if don't exist or if 'dir' is None
    if dir is None:
       create_directory("hyperparameter_test/")
       self.__dir = "hyperparameter_test/"
    else: self.__dir = dir

    if not os.path.exists(self.__dir + "best_hyperparameter.json"):
      with open(self.__dir + "best_hyperparameter.json", 'w', encoding='utf-8') as file:
        pass

    # Param optimization
    self.__pruner = HyperbandPruner(min_resource=min_resource,
                                    max_resource=max_resource, 
                                    reduction_factor=reduction_factor)
    self.__study_name = study_name
    self.__direction = direction
    self.__study = create_study(study_name=self.__direction, 
                                direction=self.__direction, 
                                pruner=self.__pruner)
    
    # Param model
    self.__n_iterations = number_of_iterations
    self.__H = H
    self.__optimizer = optimizer
    self.__X_train = X_train
    self.__Y_train = Y_train
    self.__X_val = X_val
    self.__Y_val = Y_val
    self.__epochs = epochs
    self.__batch_size = batch_size
    
    # Callback config
    self.__patience = patience
    self.__verbose = verbose
    self.__restore_best_weights = restore_best_weights

    self.inser_manual_trials()
  
  def set_training_data(self, H: int, 
                        optimizer: str = 'Adam', 
                        X_train: np.ndarray = None, 
                        Y_train: np.ndarray = None, 
                        X_val: np.ndarray = None, 
                        Y_val: np.ndarray = None, 
                        epochs: int = 20, 
                        batch_size: int = 32) -> None:
    self.__H = H
    self.__optimizer = optimizer
    self.__X_train = X_train
    self.__Y_train = Y_train
    self.__X_val = X_val
    self.__Y_val = Y_val
    self.__epochs = epochs
    self.__batch_size = batch_size

  def inser_manual_trials(self):
    pass

  def importance(self):
    param_importances = get_param_importances(self.__study)

    print("********** Importance of hyperparameters: **********")
    for param, importance in param_importances.items():
        print(f"  {param}: {importance:.8f}")
  
  def save(self, filename: str = None):
    if filename is None: filename = self.__study_name
    with open(filename + '.hyp', 'w') as custom_file:
      best_params = self.__study.best_params
      for clave, valor in best_params.items():
          custom_file.write(f"{clave}: {valor}\n")

  def new_study(self, study_name: str = "New_Study", direction: str = None):
     self.__study = create_study(study_name=study_name, 
                                 direction=direction, 
                                 pruner=self.__pruner)

  def objective(self, trial):
      clear_session()
      # Hiperparametros variables
      num_stacks = trial.suggest_int('num_stacks', 1, 15)
      heads = trial.suggest_int('heads', 1, 15)
      epsilon = trial.suggest_float('epsilon', 1e-8, 1e-4, log=True)
      num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 1, 50)
      num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 1, 50)
      list_neuron_feed = [trial.suggest_int(f'list_neuron_feed_{i}', 1, 80) for i in range(num_neurons_layers_feed)]
      list_neuron_hide = [trial.suggest_int(f'list_neuron_hide_{i}', 1, 80) for i in range(num_neurons_layers_hide)]

      model = create_model(heads=heads,
                          H=self.__H,
                          optimizer=self.__optimizer,
                          epsilon=epsilon,
                          num_stacks=num_stacks,
                          list_neuron_feed=list_neuron_feed,
                          list_neuron_hide=list_neuron_hide)

      pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
      
      '''
      early_stopping_callback = EarlyStopping(
          monitor='val_loss',  
          patience=self.__patience,         
          mode='min',          
          verbose=self.__verbose,          
          restore_best_weights=self.__restore_best_weights
      )
      '''
      best_hyp_saver = BestHyperparameterSaver(trial, 
                                               monitor='val_loss', filename=self.__dir + "best_hyperparameter.json")
      
      history = model.fit(x=self.__X_train, y=self.__Y_train, 
                          epochs=self.__epochs, 
                          batch_size=self.__batch_size, 
                          verbose=self.__verbose, 
                          validation_data=(self.__X_val, self.__Y_val),
                          callbacks=[pruning_callback, 
                                     early_stopping_callback,
                                     best_hyp_saver])
      
      
      val_loss = history.history['val_loss'][-1]

      return val_loss
  
  def optimize(self):
    self.__study.optimize(func=self.objective, n_trials=self.__n_iterations)