from keras.api.callbacks import Callback
from keras.api.metrics import Metric
import json
import os

class HyperparameterSaver(Callback):
    def __init__(self, 
                 trial, 
                 filename:str="hyperparameters.json",
                 monitor:str = 'val_loss',
                 metrics:list[str | Metric] = None):
        super(HyperparameterSaver, self).__init__()
        self.trial = trial
        self.filename = filename
        self.monitor = monitor
        self.all_hyperparameters = self.__open_file(self.filename)
        self.metrics = []
        if metrics is not None:
            for metric in metrics:
                if isinstance(metric, str):
                    self.metrics.append('val_' + metric)
                elif isinstance(metric, Metric) or issubclass(metric, Metric):
                    self.metrics.append('val_' + metric.name)


    def on_epoch_end(self, epoch, logs=None):
        hyperparameters = self.trial.params.copy()

        new_dictionary = {}

        new_dictionary['epoch'] = epoch
        new_dictionary[self.monitor] = logs.get(self.monitor)

        if self.metrics is not None:
            for metric in self.metrics:
                new_dictionary[metric] = logs.get(metric)

        new_dictionary['num_stacks'] = hyperparameters['num_stacks']
        new_dictionary['num_heads'] = hyperparameters['num_heads']
        new_dictionary['epsilon'] = hyperparameters['epsilon']
        new_dictionary['key_dim'] = hyperparameters['key_dim']

        new_dictionary['value_dim'] = hyperparameters['value_dim']

        new_dictionary['dropout'] = hyperparameters['dropout']
        new_dictionary['rate'] = hyperparameters['rate']
        new_dictionary['activation_hide'] = hyperparameters['activation_hide']
        new_dictionary['activation_feed'] = hyperparameters['activation_feed']

        new_dictionary['n_dropout_hide'] = hyperparameters['n_dropout_hide']
        new_dictionary['n_dropout_feed'] = hyperparameters['n_dropout_feed']
        
        num_neurons_layers_feed = hyperparameters['num_neurons_layers_feed']
        num_neurons_layers_hide = hyperparameters['num_neurons_layers_hide']

        neurons_feed = [hyperparameters[f'list_neurons_feed_{i}'] for i in range(num_neurons_layers_feed)]
        neurons_hide = [hyperparameters[f'list_neurons_hide_{i}'] for i in range(num_neurons_layers_hide)]

        new_dictionary['list_neurons_feed'] = neurons_feed
        new_dictionary['list_neurons_hide'] = neurons_hide

        self.all_hyperparameters = self.__open_file(self.filename)

        if len(self.all_hyperparameters) == 0: 
            self.all_hyperparameters.append(new_dictionary)
        else:
            aux_index = self.__is_there(new_dictionary=new_dictionary)
            if aux_index != -1:
                if new_dictionary[self.monitor] < self.all_hyperparameters[aux_index][self.monitor]:
                    self.all_hyperparameters[aux_index] = new_dictionary
            else:
                self.all_hyperparameters.append(new_dictionary)

        with open(self.filename, 'w') as file:
            json.dump(self.all_hyperparameters, file, indent=4)

    def __is_there(self, new_dictionary):
        for index in range(len(self.all_hyperparameters)):
            if self.__is_equals(self.all_hyperparameters[index], new_dictionary, exceptions=['epoch', self.monitor] + self.metrics) == True:
                return index
        return -1
    
    def __is_equals(self, dictionary1, dictionary2, exceptions = None):
        if exceptions is None:
            exceptions = []
        
        dic1 = {k: v for k, v in dictionary1.items() if k not in exceptions}
        dic2 = {k: v for k, v in dictionary2.items() if k not in exceptions}

        return dic1 == dic2
    
    def __open_file(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                try:
                    content = json.load(file)
                except json.JSONDecodeError:
                    print("JSON file is corrupt.")
                    content = []
        else:
            content = []
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(content, file, indent=4)
        
        return content

    
    
