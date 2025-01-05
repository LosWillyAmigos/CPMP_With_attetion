from cpmp_ml.utils.generator import load_simbol
import json
import numpy as np
import uuid

def save_data_json(States : np.ndarray, Labels : np.ndarray, name_file : str, verbose: bool = True) -> bool:
    """
    Guarda un arreglo de matrices y un arreglo de arreglos en formato JSON.

    :param States: Arreglo de matrices a guardar.
    :param Labels: Arreglo de arreglos a guardar.
    :param name_file: Nombre del archivo JSON (sin extensión).
    """
    datos = []
    size = len(States)

    for idx, (matrices, labels) in enumerate(zip(States, Labels)):
        element = {
            "_id": str(uuid.uuid4()),  # Generar un nuevo ID único
            "State": matrices.tolist(),
            "Labels": labels.tolist()
        }
        datos.append(element)

        if verbose: load_simbol(idx + 1, size, text= 'Datos guardados: ')

    with open(name_file + '.json', 'w') as archivo_json:
        json.dump(datos, archivo_json, indent=2)

    return True

def load_data_json(name_file: str, verbose: bool = True) -> tuple:
    with open(name_file, 'r') as archivo_json:
        data = json.load(archivo_json)

    states, labels = [], []
    data_size = len(data)
    cont = 0

    for input in data:
        states.append(np.array(input.get("State", [])))
        labels.append(np.array(input.get("Labels", [])))

        cont += 1
        if verbose: load_simbol(cont, data_size, text= 'Datos cargados: ')

    return np.stack(states), np.stack(labels)

def load_data_from_json(file_path: str, verbose: bool = True) -> dict:
    """
    The purpose of this function is to load data from a JSON file.

    Input:
        file_path: The path to the JSON file containing the data.
    
    Returns:
        data: A dictionary structured as described in the original function.
    """
    data = dict()

    # Load JSON data from the specified file
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    data_size = len(json_data)
    cont = 0
    # Process the data
    for states in json_data:
        state_len = str(len(states['States']))
        if state_len not in data:
            data[state_len] = {'States': [states['States']], 'Labels': [states['Labels']]}
        else:
            data[state_len]['States'].append(states['States'])
            data[state_len]['Labels'].append(states['Labels'])
        
        cont += 1
        if verbose: load_simbol(cont, data_size, text= 'Datos cargados: ')

    return data
