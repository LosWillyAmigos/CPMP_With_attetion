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
