from attentional_cpmp.model import create_model

from optuna.visualization import plot_param_importances
from optuna.importance import get_param_importances
from optuna.pruners import HyperbandPruner
from optuna import create_study
from keras.backend import clear_session

import numpy as np

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
                number_of_iterations: int = 10) -> None:
    self.__pruner = HyperbandPruner(min_resource=min_resource,
                                    max_resource=max_resource, 
                                    reduction_factor=reduction_factor)
    self.__study = create_study(study_name=study_name, 
                                direction=direction, 
                                pruner=self.__pruner)
    self.__n_iterations = number_of_iterations

    self.__study_name = study_name

    self.__H = H
    self.__optimizer = optimizer
    self.__X_train = X_train
    self.__Y_train = Y_train
    self.__X_val = X_val
    self.__Y_val = Y_val
    self.__epochs = epochs
    self.__batch_size = batch_size
    
    self.inser_manual_trials()

  def objective(self, trial):
      clear_session()

      # Hiperparametros variables
      num_stacks = trial.suggest_int('num_stacks', 1, 15)
      heads = trial.suggest_int('heads', 1, 15)
      epsilon = trial.suggest_float('epsilon', 1e-8, 1e-4, log=True)
      num_neurons_layers_feed = trial.suggest_int('num_neurons_layers_feed', 1, 50)
      num_neurons_layers_hide = trial.suggest_int('num_neurons_layers_hide', 1, 50)
      list_neuron_feed = [trial.suggest_int(f'list_neuron_feed_{i}', 1, 100) for i in range(num_neurons_layers_feed)]
      list_neuron_hide = [trial.suggest_int(f'list_neuron_hide_{i}', 1, 100) for i in range(num_neurons_layers_hide)]

      model = create_model(heads=heads,
                          H=self.__H,
                          optimizer=self.__optimizer,
                          epsilon=epsilon,
                          num_stacks=num_stacks,
                          list_neuron_feed=list_neuron_feed,
                          list_neuron_hide=list_neuron_hide)

      history = model.fit(self.__X_train, self.__Y_train, 
                          epochs=self.__epochs, 
                          batch_size=self.__batch_size, 
                          verbose=0, 
                          validation_data=(self.__X_val, self.__Y_val))
      
      clear_session()
      val_mse = history.history['mse'][-1]

      return val_mse
  
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

  def optimize(self):
    self.__study.optimize(func=self.objective, n_trials=self.__n_iterations)

  def importance(self):
    param_importances = get_param_importances(study)

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