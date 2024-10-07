from keras.callbacks import Callback
import json

class HyperparameterSaver(Callback):
    def __init__(self, 
                 trial, 
                 filename="hyperparameters.json"):
        super(HyperparameterSaver, self).__init__()
        self.trial = trial
        self.filename = filename
        self.all_hyperparameters = []

    def on_epoch_end(self, epoch, logs=None):
        # Recoger los hiperparámetros actuales
        hyperparameters = self.trial.params.copy()
        hyperparameters['epoch'] = epoch
        hyperparameters['val_loss'] = logs.get('val_loss')
        hyperparameters['val_accuracy'] = logs.get('val_accuracy')
        
        # Añadir los hiperparámetros a la lista
        self.all_hyperparameters.append(hyperparameters)

        # Guardar los hiperparámetros en el archivo
        with open(self.filename, 'w') as f:
            json.dump(self.all_hyperparameters, f, indent=4)
    