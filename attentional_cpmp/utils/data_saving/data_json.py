import json
import numpy as np
import uuid

def save_data_json(States : np.array, Labels : np.array, name_file : str) -> bool:
    """
    Guarda un arreglo de matrices y un arreglo de arreglos en formato JSON.

    :param States: Arreglo de matrices a guardar.
    :param Labels: Arreglo de arreglos a guardar.
    :param name_file: Nombre del archivo JSON (sin extensión).
    """
    datos = []

    for matrices, labels in zip(States, Labels):
        element = {
            "_id": str(uuid.uuid4()),  # Generar un nuevo ID único
            "State": matrices.tolist(),
            "Labels": labels.tolist()
        }
        datos.append(element)

    with open(name_file + '.json', 'w') as archivo_json:
        json.dump(datos, archivo_json, indent=2)

    return True

def load_data_json(name_file:str) -> tuple:
    with open(name_file + '.json', 'r') as archivo_json:
        data = json.load(archivo_json)

    states = []
    labels = []

    for input in data:
        states.append(np.array(input.get("State", [])))
        labels.append(np.array(input.get("Labels", [])))


    return np.stack(states), np.stack(labels)
