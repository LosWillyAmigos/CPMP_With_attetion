from cpmp_ml.utils.generator import load_simbol
import numpy as np
import pymongo
import re

import pymongo.collection
import pymongo.errors

def connect_to_server(uri: str) -> pymongo.MongoClient:
    """
    The purpose of this function is to establish 
    a connection between the MongoDB server and the program.

    Input:
        uri (string): The URL of the MongoDB server.
    """
    try: 
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS= 10000)
        client.server_info()
        print('Conexión exitosa con el servidor de MongoDB\n')

        return client
    
    except pymongo.errors.ServerSelectionTimeoutError as identifier:
        patron = r"Timeout: ([\d\.]+)s"
        time = re.search(patron, str(identifier))

        print(f'Tiempo de espera excedido: {time.group(1)}\n')
    except pymongo.errors.ConnectionFailure as conection_Error:
        print('Error al conectarse a mongodb: ' + str(conection_Error))
    except pymongo.errors.OperationFailure as auth_error:
        print('Error de autenticación!')
    
    return None

def load_data_mongo(collection: pymongo.collection.Collection, verbose: bool = True) -> dict:
    """
    The purpose of this function is to load data from MongoDB.

    Input:
        collection: The MongoDB client's database from which 
                    to load the data.
    """
    try:
        data = dict()
        collection_size = collection.count_documents({})
        cont = 0

        for states in collection.find():
            if str(len(states['States'])) not in data:
                data.update({str(len(states['States'])): {'States': [states['States']], 'Labels': [states['Labels']]}})
            else:
                data[str(len(states['States']))]['States'].append(states['States'])
                data[str(len(states['States']))]['Labels'].append(states['Labels'])
            cont += 1

            if verbose: load_simbol(cont, collection_size, text= 'Datos cargados: ')

        return data
    except pymongo.errors.ConnectionFailure as conection_Error:
        print('Error en la conexión con la base de datos: ' + str(conection_Error))
        return None

def load_data_mongo_2(collection: pymongo.collection.Collection, verbose: bool = False) -> tuple:
    try:
        data, labels = [], []
        collection_size = collection.count_documents({})
        cont = 0

        for states in collection.find():
            data.append(np.array(states['States']))
            labels.append(np.array(states['Labels']))

            cont += 1
            if verbose: load_simbol(cont, collection_size, text= 'Datos cargados: ')

        return data, labels
    
    except pymongo.errors.ConnectionFailure as conection_Error:
        print('Error en la conexión con la base de datos: ' + str(conection_Error))
        return None, None

def save_data_mongo(collection, data: list[np.ndarray], labels: list[np.ndarray], verbose: bool = True) -> bool:
    """
    The purpose of this function is to store all states 
    and labels for the CPMP model in a MongoDB database.

    Input: 
        collection: The MongoDB client's database from which 
                    to load the data.
        data (list): List containing the states of the CPMP 
                     problem.
        labels (list): List containing the labels of the CPMP
                       problem.
    """
    size = len(data)

    for i in range(size):
        try:
            if isinstance(data[i], np.ndarray) and isinstance(labels[i], np.ndarray):
                state = {'States': data[i].tolist(), 'Labels': labels[i].tolist()}
            elif isinstance(data[i], np.ndarray) and not isinstance(labels[i], np.ndarray):
                state = {'States': data[i].tolist(), 'Labels': labels[i]}
            elif not isinstance(data[i], np.ndarray) and isinstance(labels[i], np.ndarray):
                state = {'States': data[i], 'Labels': labels[i].tolist()}
                
            collection.insert_one(state)

            if verbose: load_simbol(i + 1, size, text= 'Datos guardados: ')
        except pymongo.errors.ConnectionFailure as conection_Error:
            print('Error en la conexión con la base de datos: ' + str(conection_Error))
            return False
    
    return True