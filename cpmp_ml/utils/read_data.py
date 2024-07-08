import numpy as np
import pymongo

def connect_to_server(uri):
    """
    The purpose of this function is to establish 
    a connection between the MongoDB server and the program.

    Input:
        uri (string): The URL of the MongoDB server.
    """
    try: 
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS= 1000)
        client.server_info()
        print('Conection Success')

        return client
    
    except pymongo.errors.ServerSelectionTimeoutError as identifier:
        print('tiempo excedido' + identifier)

    except pymongo.errors.ConnectionFailure as conection_Error:
        print('Error al conectarse a mongodb' + conection_Error)

def load_data_mongo(collection):
    """
    The purpose of this function is to load data from MongoDB.

    Input:
        collection: The MongoDB client's database from which 
                    to load the data.
    """
    data = []
    labels = []

    for states in collection.find():
        data.append(states['State'])
        labels.append(states['Labels'])
    
    return np.stack(data), np.stack(labels)

def save_data_mongo(collection, data: list, labels: list):
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
            state = {'State': data[i].tolist(), 'Labels': labels[i].tolist()}
            collection.insert_one(state)
        except pymongo.errors.ConnectionFailure as conection_Error:
            print('Error al conectarse a mongodb' + conection_Error)

def load_data_csv(name):
    data = []
    labels_1 = []

    with open(name + '.csv', 'r') as archivo:
        total = int(archivo.readline().split(':')[1])
        line = archivo.readline().split(':')
        size_stacks = int(line[1].split(',')[0])
        size_height = int(line[2])
        archivo.readline()

        for i in range(total):
            matrix = archivo.readline().split(':')[1].split(',')
            matrix = np.array(matrix, dtype= float)
            matrix = np.reshape(matrix, (size_stacks, size_height))

            label_1 = archivo.readline().split(':')[1].split(',')
            label_1 = np.array(label_1, dtype= float)

            data.append(matrix)
            labels_1.append(label_1)

            archivo.readline()

    return np.stack(data), np.stack(labels_1)