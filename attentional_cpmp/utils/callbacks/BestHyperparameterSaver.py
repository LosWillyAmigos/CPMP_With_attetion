from keras.callbacks import Callback
import json

class BestHyperparameterSaver(Callback):
    def __init__(self, 
                 trial, 
                 monitor='val_loss', 
                 mode='min', 
                 filename="best_hyperparameters.json"):
        super(BestHyperparameterSaver, self).__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = None
        self.best_hyperparameters = None

        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if self.__is_better(current_score):
            self.best_score = current_score
            self.best_hyperparameters = self.trial.params.copy()
            self.best_hyperparameters['epoch'] = epoch
            self.best_hyperparameters[self.monitor] = current_score

            with open(self.filename, 'w') as f:
                json.dump(self.best_hyperparameters, f)

    def __is_better(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score
        else:
            return current_score > self.best_score