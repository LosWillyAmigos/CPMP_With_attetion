from keras.callbacks import Callback
import json

class HyperparameterSaver(Callback):
    def __init__(self, 
                 trial, 
                 filename="hyperparameters.json",
                 monitor:str = 'val_loss'):
        super(HyperparameterSaver, self).__init__()
        self.trial = trial
        self.filename = filename
        self.monitor = monitor
        self.all_hyperparameters = []

    def on_epoch_end(self, epoch, logs=None):
        # Recoger los hiperparámetros actuales
        hyperparameters = self.trial.params.copy()
        
        hyperparameters['epoch'] = epoch
        hyperparameters[self.monitor] = logs.get(self.monitor)

        neurons_feed = [hyperparameters[f'list_neurons_feed_{i}'] for i in range(hyperparameters['num_neurons_layers_feed'])]
        neurons_hide = [hyperparameters[f'list_neurons_hide_{i}'] for i in range(hyperparameters['num_neurons_layers_hide'])]

        hyperparameters['list_neurons_feed'] = neurons_feed
        hyperparameters['list_neurons_hide'] = neurons_hide

        # Añadir los hiperparámetros a la lista
        self.all_hyperparameters.append(hyperparameters)

        # Guardar los hiperparámetros en el archivo
        with open(self.filename, 'w') as f:
            json.dump(self.all_hyperparameters, f, indent=4)
    
    
