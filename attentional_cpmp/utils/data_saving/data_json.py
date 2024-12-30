#from cpmp_ml.utils.generator import load_simbol
#from cpmp_ml.utils import delete_terminal_lines
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

        '''
        if verbose:
            load_simbol(idx + 1, size, text= 'Datos guardados:')
            if idx + 1 < size: delete_terminal_lines(1)
        '''

    with open(name_file + '.json', 'w') as archivo_json:
        json.dump(datos, archivo_json, indent=2)

    return True

def load_data_json(name_file:str) -> tuple:
    with open(name_file, 'r') as archivo_json:
        data = json.load(archivo_json)

    states = []
    labels = []

    for input in data:
        states.append(np.array(input.get("State", [])))
        labels.append(np.array(input.get("Labels", [])))

    return np.stack(states), np.stack(labels)

def load_data_from_json(file_path):
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

    # Process the data
    for states in json_data:
        state_len = str(len(states['States']))
        if state_len not in data:
            data[state_len] = {'States': [states['States']], 'Labels': [states['Labels']]}
        else:
            data[state_len]['States'].append(states['States'])
            data[state_len]['Labels'].append(states['Labels'])

    return data
