import os
import numpy as np
import random

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_data(data, key):
    # Obtener el diccionario para la clave específica
    subset = data.get(key)
    if not subset:
        raise KeyError(f"No existe la clave {key} en el diccionario.")

    states = subset["States"]
    labels = subset["Labels"]

    return np.stack(states), np.stack(labels)

def split_data(data, key, percentage):
    # Obtener el diccionario para la clave específica
    subset = data.get(key)
    if not subset:
        raise KeyError(f"No existe la clave {key} en el diccionario.")

    states = subset["States"]
    labels = subset["Labels"]

    # Determinar la cantidad de elementos a extraer
    sample_size = int(len(states) * percentage)

    # Obtener índices aleatorios para la muestra
    indices = random.sample(range(len(states)), sample_size)

    # Crear subconjuntos para el 20% y el 80%
    sampled_states = [states[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]

    remaining_states = [states[i] for i in range(len(states)) if i not in indices]
    remaining_labels = [labels[i] for i in range(len(labels)) if i not in indices]

    return (np.stack(sampled_states), np.stack(sampled_labels)), (np.stack(remaining_states), np.stack(remaining_labels))

def get_hyperparams(dictionary):
    '''
    Función para obtener los hiperparámetros del modelo a partir de un diccionario,
    esto es útil puesto que un diccionario puede tener más elementos que los hiperparámetros.

    Args:
        dictionary (dict): Diccionario con los hiperparámetros del modelo.

    Returns:
        dict: Diccionario con los hiperparámetros del modelo.
    '''
    new_dictionary = {}

    new_dictionary['num_stacks'] = dictionary['num_stacks']
    new_dictionary['num_heads'] = dictionary['num_heads']
    new_dictionary['epsilon'] = dictionary['epsilon']
    new_dictionary['key_dim'] = dictionary['key_dim']
    new_dictionary['value_dim'] = dictionary['value_dim']
    new_dictionary['dropout'] = dictionary['dropout']
    new_dictionary['rate'] = dictionary['rate']
    new_dictionary['activation_hide'] = dictionary['activation_hide']
    new_dictionary['activation_feed'] = dictionary['activation_feed']

    new_dictionary['n_dropout_hide'] = dictionary['n_dropout_hide']
    new_dictionary['n_dropout_feed'] = dictionary['n_dropout_feed']
    new_dictionary['list_neurons_feed'] = dictionary['list_neurons_feed']
    new_dictionary['list_neurons_hide'] = dictionary['list_neurons_hide']

    return new_dictionary

def get_config_model(dictionary):
    
    '''
    Función para obtener los hiperparámetros y la configuración del modelo a partir de un diccionario.

    Args:
        dictionary (dict): Diccionario con los hiperparámetros y la configuración del modelo.
    
    Returns:
        dict: Diccionario con los hiperparámetros y la configuración del modelo.
    '''

    new_dictionary = {}
    new_dictionary['H'] = dictionary['H']
    new_dictionary['metrics'] = dictionary['metrics']
    new_dictionary['optimizer'] = dictionary['optimizer']
    new_dictionary['loss'] = dictionary['loss']

    new_dictionary['num_stacks'] = dictionary['num_stacks']
    new_dictionary['num_heads'] = dictionary['num_heads']
    new_dictionary['epsilon'] = dictionary['epsilon']
    new_dictionary['key_dim'] = dictionary['key_dim']
    new_dictionary['value_dim'] = dictionary['value_dim']
    new_dictionary['dropout'] = dictionary['dropout']
    new_dictionary['rate'] = dictionary['rate']
    new_dictionary['activation_hide'] = dictionary['activation_hide']
    new_dictionary['activation_feed'] = dictionary['activation_feed']

    new_dictionary['n_dropout_hide'] = dictionary['n_dropout_hide']
    new_dictionary['n_dropout_feed'] = dictionary['n_dropout_feed']
    new_dictionary['list_neurons_feed'] = dictionary['list_neurons_feed']
    new_dictionary['list_neurons_hide'] = dictionary['list_neurons_hide']

    return new_dictionary