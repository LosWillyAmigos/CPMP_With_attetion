from cpmp_ml.generators import generate_data_v2
from cpmp_ml.generators import generate_data_v3
from cpmp_ml.optimizer import GreedyV1
from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import AttentionModel
from attentional_cpmp.utils import connect_to_server
from attentional_cpmp.utils import load_data_mongo
from attentional_cpmp.utils import save_data_mongo
from dotenv import load_dotenv
import numpy as np
import getpass
import os

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

MONGO_URI = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/?authSource={DB_NAME}'
MODEL_ROUTE = './models/'

def get_os():
    if os.name == 'nt':
        return 'Windows'
    elif os.name == 'posix':
        return 'Linux/MacOS'
    return 'Unknown'

def try_again():
    act = input('Quiere volver a intentar? (S / N) ').lower()
                
    if sys == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

    if act == 's': return True
    return False

def dis_menu_gen1():
    print('|*|*************| Generador V1 |*************|*|')
    print('|*| Los datos que entregará este generador   |*|')
    print('|*| será el estado inicial de un problema    |*|')
    print('|*| con su solución.                         |*|')
    print('|*|**********| CPMP_With_Attention |*********|*|')

def dis_menu_gen2():
    return 

def dis_menu_gen3():
    return 

def dis_menu_change_model_route():
    return

def dis_menu_change_uri_mongo():
    return

def dis_load_data_mongo():
    print('|*|*************| MongoDB |*************|*|')
    print('|*| Este menú le dará apoyo para        |*|')
    print('|*| almacenar los datos generados en    |*|')
    print('|*| MongoDB.                            |*|')
    print('|*|*******| CPMP_With_Attention |*******|*|')

def dis_load_data():
    print('|*|*************| Guardado de datos |*************|*|')
    print('|*| Como desea almacenar sus datos?               |*|')
    print('|*| 1) Formato JSON                               |*|')
    print('|*| 2) Cluster MongoDB                            |*|')
    print('|*|************| CPMP_With_Attention |************|*|')

def dis_main_menu():
    print('|*|*************| Menú del generador |*************|*|')
    print('|*|  1) Generar datos V1                           |*|')
    print('|*|  2) Generar datos V2                           |*|')
    print('|*|  3) Generar datos V3                           |*|')
    print('|*|  4) Cambiar ruta del modelo para generador V3  |*|')
    print('|*|  5) Cambiar URI de MongoDB                     |*|')
    print('|*|  6) Salir                                      |*|')
    print('|*|************| CPMP_With_Attention |*************|*|')

def change_mongodb_data():
    global DB_HOST
    global DB_USER
    global DB_PASSWORD
    global DB_NAME
    global MONGO_URI
    
    DB_HOST = input('Indique la IP del servidor mongodb: ')
    DB_USER = input('Indique su nombre de usuario: ')
    DB_PASSWORD = getpass.getpass('Introduce tu contraseña: ')
    DB_NAME = input('Ingrese el nombre de la base de datos: ')

    MONGO_URI = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/?authSource={DB_NAME}'
    print('Datos cambiados con éxito!')

def load_data_mongo_menu():
    global MONGO_URI

    while True:
        dis_load_data_mongo()

        data_base_name = input('Indique el nombre de la base de datos: ')
        collection_name = input('Indique el nombre de la colección: ')

        client = connect_to_server(MONGO_URI)
        if client is None and try_again():
            change_mongodb_data()
            continue

        data_base = client[data_base_name]

        print('Iniciando Guardado...')
        save_data_mongo(data_base[collection_name], data, labels)
        
        print('Guardado exitoso!')
        client.close()

def load_data_menu(data: list, labels: list):
    while True:
        dis_load_menu()

        try:
            option = int(input('Escoja una opción:'))
        except ValueError:
            print('Coloque una opción valida (1 - 2)')
            input('Pulse Enter para continuar')

            if sys == 'Windows':
                os.system('cls')
            else:
                os.system('clear')

            continue
        except KeyboardInterrupt:
            print('')
            print('Cerrando el programa...')
            exit()
        
        if option == 1:
            continue
        if option == 2:


def generate_v1_menu(sys: str):
    while True:
        dis_menu_gen1()

        S = input('Cuantos stacks contiene el problema? ')
        H = input('De que altura es el problema? ')
        N = input('Cuantos contenedores posee el problema? ')
        perms_by_layout = input('Cuantas permutaciones necesita? (n ≥ 1) ')
        max_steps = input('En cuantos pasos como máximo se debe resolver el problema? (n ≥ 1) ')
        sample_size = input('Cuantos datos necesita en total? ')
        optimizer = input('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ').lower()
        adapter = input('Que adaptador necesitas? (AttentionModel, LinealModel)').lower()
        act = input('Está seguro de sus elecciones? (S / N) ').lower()

        if act == 'n':
            if sys == 'Windows':
                os.system('cls')
            else:
                os.system('clear')
            return
        
        elif act == 's':
            if S.isdigit(): S = int(S)
            else:
                print('La cantidad de stacks escogida no es un número.')
                if try_again(): continue
                else: return

            if H.isdigit(): H = int(H)
            else:
                print('La altura escogida no es un número.')
                if try_again(): continue
                else: return

            if N.isdigit(): N = int(N)
            else:
                print('La cantidad de contenedores escogidos no es un número.')
                if try_again(): continue
                else: return

            if sample_size.isdigit(): sample_size = int(sample_size)
            else:
                print('La cantidad de datos escogidos no es un número.')
                if try_again(): continue
                else: return

            if perms_by_layout.isdigit() and int(perms_by_layout) >= 1: perms_by_layout = int(perms_by_layout)
            else:
                print('La cantidad de permutaciones escogidos no es un número')
                print('o no cumple con los parámetros exigidos.')
                if try_again(): continue
                else: return

            if max_steps.isdigit() and int(max_steps) >= 1: max_steps = int(max_steps)
            else:
                print('La cantidad de pasos escogidos no es un número')
                print('o no cumple con los parámetros exigidos.')
                if try_again(): continue
                else: return

            if optimizer == 'greedyv1': optimizer = GreedyV1()
            elif optimizer == 'greedyv2': optimizer = GreedyV2()
            elif optimizer == 'greedymodel': optimizer = GreedyModel()
            else:
                print('El optimizador escogido no existe.')
                if try_again(): continue
                else: return

            if adapter == 'attentionmodel': adapter = AttentionModel()
            elif adapter == 'linealmodel': adapter = LinealModel()
            else:
                print('El adaptador escogido no existe')
                if try_again(): continue
                else: return

            print('La generación de datos comienza!')

            data, labels = generate_data_v1(S, H, N, sample_size, verbose= True, perms_by_layout= 1, 
                                            solver= optimizer, adapter= AttentionModel(), max_steps= max_steps)

            
def main_menu():
    sys = get_os()

    while True:
        dis_main_menu()
        
        try:
            option = int(input('Escoja una opción:'))
        except ValueError:
            print('Coloque una opción valida (1 - 6)')
            input('Pulse Enter para continuar')

            if sys == 'Windows':
                os.system('cls')
            else:
                os.system('clear')
            
            continue
        except KeyboardInterrupt:
            print('')
            print('Cerrando el programa...')
            exit()

        if option == 1:
            generate_v1_menu(sys)
        elif option == 2:
            continue
        elif option == 3:
            continue
        elif option == 4:
            continue
        elif option == 5:
            continue
        elif option == 6:
            print('')
            print('Cerrando el programa...')
            exit()
        else:
            print('opción invalida, por favor colocar un valor entre el 1 al 6')

        if sys == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

if __name__ == '__main__':
    main_menu()