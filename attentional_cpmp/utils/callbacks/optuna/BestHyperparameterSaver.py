from keras.callbacks import Callback
import json
import os

class BestHyperparameterSaver(Callback):
    def __init__(self, 
                 trial, 
                 monitor:str='val_loss',
                 metrics:list = None,
                 mode:str='min', 
                 filename:str="best_hyperparameters.json"):
        super(BestHyperparameterSaver, self).__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = None
        self.best_hyperparameters = None
        self.metrics = []
        if metrics is not None:
            for metric in metrics:
                 self.metrics.append('val_' + metric)

        if self.mode == 'min':
            default_best_score = float('inf')
        else:
            default_best_score = float('-inf')

        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as file:
                    data = json.load(file)
                    self.best_score = data.get(self.monitor, default_best_score)
            except (json.JSONDecodeError, KeyError, TypeError):
                self.best_score = default_best_score
        else:
            self.best_score = default_best_score

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if self.__is_better(current_score):
            self.best_score = current_score
            self.best_hyperparameters = self.trial.params.copy()
            self.best_hyperparameters['epoch'] = epoch
            self.best_hyperparameters[self.monitor] = current_score

            with open(self.filename, 'w') as f:
                json.dump(self.__get_hyp(self.best_hyperparameters, logs), f)

    def __get_hyp(self, dict_copy_hyp, logs):
        new_dictionary = {}

        new_dictionary['epoch'] = dict_copy_hyp['epoch']
        new_dictionary[self.monitor] = dict_copy_hyp[self.monitor]
        
        for metric in self.metrics:
            new_dictionary[metric] = logs.get(metric)

        new_dictionary['num_stacks'] = dict_copy_hyp['num_stacks']
        new_dictionary['num_heads'] = dict_copy_hyp['num_heads']
        new_dictionary['epsilon'] = dict_copy_hyp['epsilon']
        new_dictionary['key_dim'] = dict_copy_hyp['key_dim']

        new_dictionary['value_dim'] = dict_copy_hyp['value_dim']

        new_dictionary['dropout'] = dict_copy_hyp['dropout']
        new_dictionary['rate'] = dict_copy_hyp['rate']
        new_dictionary['activation_hide'] = dict_copy_hyp['activation_hide']
        new_dictionary['activation_feed'] = dict_copy_hyp['activation_feed']

        new_dictionary['n_dropout_hide'] = dict_copy_hyp['n_dropout_hide']
        new_dictionary['n_dropout_feed'] = dict_copy_hyp['n_dropout_feed']
        
        num_neurons_layers_feed = dict_copy_hyp['num_neurons_layers_feed']
        num_neurons_layers_hide = dict_copy_hyp['num_neurons_layers_hide']

        neurons_feed = [dict_copy_hyp[f'list_neurons_feed_{i}'] for i in range(num_neurons_layers_feed)]
        neurons_hide = [dict_copy_hyp[f'list_neurons_hide_{i}'] for i in range(num_neurons_layers_hide)]

        new_dictionary['list_neurons_feed'] = neurons_feed
        new_dictionary['list_neurons_hide'] = neurons_hide

        return new_dictionary

    def __is_better(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score
        else:
            return current_score > self.best_score