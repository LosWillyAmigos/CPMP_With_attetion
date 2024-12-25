import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from cpmp_ml.generators import generate_data_v1
from cpmp_ml.generators import generate_data_v2
from cpmp_ml.generators import generate_data_v3
from cpmp_ml.optimizer import GreedyV1
from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.utils.adapters import LinealModel
from attentional_cpmp.utils import connect_to_server
from attentional_cpmp.utils import save_data_mongo
from attentional_cpmp.utils import save_data_json
from attentional_cpmp.model import load_cpmp_model
from keras.api.models import Model
from dotenv import load_dotenv
import numpy as np
import getpass
import sys

SYSTEM = os.name

def delete_terminal_lines(lines: int):
    for _ in range(lines):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

def clear_terminal():
    if SYSTEM == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def write_environment(db_host: str, db_user: str, db_password: str, db_name: str, model_route: str, json_data_route: str):
    file = open('.env', 'w', encoding= 'utf-8')

    file.write(f'# Datos de conexión a la base de datos\n')
    file.write(f'DB_HOST="{db_host}"\n')
    file.write(f'DB_USER="{db_user}"\n')
    file.write(f'DB_PASSWORD="{db_password}"\n')
    file.write(f'DB_NAME="{db_name}"\n')
    file.write(f'\n')
    file.write(f'# Ubicación de archivos\n')
    file.write(f'MODEL_ROUTE="{model_route}"\n')
    file.write(f'JSON_DATA_ROUTE="{json_data_route}"\n')

    file.close()

def install_environment():
    second_error = False
    db_host = ''
    db_user = ''
    db_password = ''
    db_name = ''
    model_route = os.getcwd().replace('\\', '/') + '/models/'
    json_data_route = os.getcwd().replace('\\', '/') + '/data/'
    
    while True:
        clear_terminal()
        while True:
            opt = input('Usted usa MongoDB? (S/N) ').lower()
            if opt in ['s', 'n']: 
                second_error = False
                break
            
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True
            
        if opt == 's':    
            db_host = input('Indique la IP del servidor mongodb (IP:Port): ')
            db_user = input('Indique su nombre de usuario: ')
            db_password = getpass.getpass('Introduce tu contraseña: ')
            db_name = input('Ingrese el nombre de la base de datos de autenticación: ')
            db_uri = f'mongodb://{db_user}:{"*" * (len(db_password) + 1)}@{db_host}/?authSource={db_name}'
            
            print('MongoDB URI:', db_uri)
            while True:
                opt = input('Está seguro de los datos ingresados? (S/N) ')
                if opt.lower() in ['s', 'n']:
                    second_error = False
                    break

                if not second_error: delete_terminal_lines(1)
                else: delete_terminal_lines(2)
                print('Por favor, coloque una opción válida.')
                second_error = True

            if opt.lower() == 'n': continue
        
        print(f'La ruta de los modelos será: {model_route}')
        while True:
            opt = input('Está seguro de la ruta de los modelos? (S/N) ').lower()
            if opt in ['s', 'n']:
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True

        if opt == 'n': model_route = input('Indique la ruta de los modelos: ')
        
        print(f'La ruta de los datos generados por JSON será: {json_data_route}')
        while True:
            opt = input('Está seguro de la ruta de los datos generados por JSON? (S/N) ').lower()
            if opt.lower() in ['s', 'n']:
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True
        
        if opt == 'n': json_data_route = input('Indique la ruta de los datos generados por JSON: ')

        write_environment(db_host, db_user, db_password, db_name, model_route, json_data_route)

        return json_data_route, model_route

def install_program():
    json_route, model_route = install_environment()
    if not os.path.exists(model_route): os.makedirs(model_route)
    if not os.path.exists(json_route): os.makedirs(json_route)

if not os.path.exists('.env'): install_program()    

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

MONGO_URI = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/?authSource={DB_NAME}'
MODEL_ROUTE = os.environ.get("MODEL_ROUTE")
JSON_DATA_ROUTE = os.environ.get("JSON_DATA_ROUTE")

def try_again():
    act = input('Quiere volver a intentar? (S / N) ').lower()

    delete_terminal_lines(1)

    if act == 's': return True
    return False

def dis_menu_gen1():
    print('|*|*************| Generador V1 |*************|*|')
    print('|*| Los datos que entregará este generador   |*|')
    print('|*| será el estado inicial de un problema    |*|')
    print('|*| con su solución.                         |*|')
    print('|*|**********| CPMP_With_Attention |*********|*|')
    print('')

def dis_menu_gen2():
    print('|*|*************| Generador V2 |*************|*|')
    print('|*| Los datos que entregará este generador   |*|')
    print('|*| serán todos los pasos para resolver un   |*|')
    print('|*| problema con sus soluciones, para esto   |*|')
    print('|*| deberá entregar un porcentaje de la      |*|')
    print('|*| de pasos que desea.                      |*|')
    print('|*|                                          |*|')
    print('|*| ESTO ES SOLO PARA MODELOS DE ATENCIÓN.   |*|')
    print('|*|**********| CPMP_With_Attention |*********|*|')
    print('')

def dis_menu_gen3():
    print('|*|*************| Generador V3 |*************|*|')
    print('|*| Los datos que entregará este generador   |*|')
    print('|*| será el estado incial de un problema     |*|')
    print('|*| con su solución, pero ampliando su       |*|')
    print('|*| búsqueda hasta un primer nivel de        |*|')
    print('|*| profundidad.                             |*|')
    print('|*|**********| CPMP_With_Attention |*********|*|')
    print('') 

def dis_menu_change_model_route():
    print('|*|*************| Ruta de los modelos |*************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los modelos.                |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_menu_change_mongodb_uri():
    print('|*|*************| URI de MongoDB |***************|*|')
    print('|*| Este menú le dará apoyo para cambiar la URI  |*|')
    print('|*| de conexión a MongoDB.                       |*|')
    print('|*|***********| CPMP_With_Attention |************|*|')
    print('')

def dis_menu_change_data_route():
    print('|*|**************| Ruta de los datos |**************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los datos generados.        |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_save_data_json():
    print('|*|*************| JSON |**************|*|')
    print('|*| Este menú le dará apoyo para      |*|')
    print('|*| almacenar los datos generados en  |*|')
    print('|*| formato JSON dentro de su unidad  |*|')
    print('|*| de almacenamiento local.          |*|')
    print('|*|******| CPMP_With_Attention |******|*|')
    print('')

def dis_save_data_mongodb():
    print('|*|*************| MongoDB |*************|*|')
    print('|*| Este menú le dará apoyo para        |*|')
    print('|*| almacenar los datos generados en    |*|')
    print('|*| MongoDB.                            |*|')
    print('|*|*******| CPMP_With_Attention |*******|*|')
    print('')

def dis_select_model():
    print('|*|***************| Selección de modelo |***************|*|')
    print('|*| Este menú le dará apoyo para seleccionar el modelo  |*|')
    print('|*| que desea utilizar para la generación de datos.     |*|')
    print('|*|***************| CPMP_With_Attention |***************|*|')
    print('')

def dis_save_data():
    print('|*|*************| Guardado de datos |*************|*|')
    print('|*| Como desea almacenar sus datos?               |*|')
    print('|*| 1) Formato JSON                               |*|')
    print('|*| 2) Cluster MongoDB                            |*|')
    print('|*|************| CPMP_With_Attention |************|*|')
    print('')

def dis_option_menu():
    print('|*|**************| Opciones |****************|*|')
    print('|*| 1) Cambiar la ruta de los modelos        |*|')
    print('|*| 2) Cambiar la URI de MongoDB             |*|')
    print('|*| 3) Cambiar la ubicación de los archivos  |*|')
    print('|*| 4) Volver al menú principal              |*|')
    print('|*|*********| CPMP_With_Attention |**********|*|')
    print('')

def dis_main_menu():
    print('|*|*************| Menú del generador |*************|*|')
    print('|*|  1) Generar datos V1                           |*|')
    print('|*|  2) Generar datos V2 (Solo para Atención)      |*|')
    print('|*|  3) Generar datos V3                           |*|')
    print('|*|  4) Opciones                                   |*|')
    print('|*|  5) Salir                                      |*|')
    print('|*|************| CPMP_With_Attention |*************|*|')
    print('')

def change_mongodb_uri():
    global DB_HOST
    global DB_USER
    global DB_PASSWORD
    global DB_NAME
    global MONGO_URI
    
    clear_terminal()
    dis_menu_change_mongodb_uri()

    DB_HOST = input('Indique la IP del servidor mongodb (IP:Port) : ')
    DB_USER = input('Indique su nombre de usuario: ')
    DB_PASSWORD = getpass.getpass('Introduce tu contraseña: ')
    DB_NAME = input('Ingrese el nombre de la base de datos de autenticación: ')

    MONGO_URI = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/?authSource={DB_NAME}'
    print('Datos cambiados con éxito!')

def save_data_mongo_menu(data: list, labels: list):
    while True:
        clear_terminal()
        dis_save_data_mongodb()
        second_error = False

        client = connect_to_server(MONGO_URI)
        if client is None and try_again():
            change_mongodb_uri()
            clear_terminal()
            continue

        while True:
            data_base_name = input('Indique el nombre de la base de datos: ')
            if data_base_name in client.list_database_names(): 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('La base de datos no existe.')
            second_error = True
        
        while True:
            collection_name = input('Indique el nombre de la colección: ')
            if collection_name in client[data_base_name].list_collection_names(): 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('La colección no existe.')
            second_error = True

        data_base = client[data_base_name]

        print('Iniciando Guardado...')
        if not save_data_mongo(data_base[collection_name], data, labels): 
            print('Error al guardar los datos.')
            input('Pulse Enter para continuar')
            
            client.close()
            
            return False
        
        print('Datos guardados con éxito!')

        client.close()

        return True

def save_data_json_menu(data: list, labels: list):
    while True:
        clear_terminal()
        dis_save_data_json()
        second_error = False
 
        file_name = input('Indique el nombre del archivo: ')
        if not os.path.exists(JSON_DATA_ROUTE + file_name + '.json'): 
            print('El archivo ya existe.')
            print('Si continua, el archivo será sobreescrito.')
            while True:   
                option = input('Desea continuar? (S / N) ').lower()
                if option in ['s', 'n']: 
                    second_error = False
                    break

                if not second_error: delete_terminal_lines(1)
                else: delete_terminal_lines(2)
                print('Por favor, coloque una opción válida.')
                second_error = True
            
            if option == 'n': return False

        print('Iniciando Guardado...')
        
        try: 
            save_data_json(data, labels, JSON_DATA_ROUTE + file_name)
        except Exception as e:
            print('Error al guardar los datos.\n')
            print(e)	
            input('Pulse Enter para continuar')

            return False

        print('Datos guardados con éxito!')

        return True

def save_data_menu(data: list, labels: list):
    while True:
        clear_terminal()
        dis_save_data()

        try:
            option = int(input('Escoja una opción: '))
        except ValueError:
            print('Coloque una opción valida (1 - 2)')
            input('Pulse Enter para continuar')

            clear_terminal()
            continue
        
        if option == 1:
            if not save_data_json_menu(data, labels) and try_again(): continue
        if option == 2:
            if not save_data_mongo_menu(data, labels) and try_again(): continue
        
        break

def get_models(route: str) -> dict:
    models = dict()
    models_names = os.listdir(route)

    if len(models_names) == 0: return None

    for i, model in enumerate(models_names):
        models.update({i + 1: model})
        print(f'{i + 1}) {model}')
    print('')

    return models

def select_model(S: int, H: int, adapter: str) -> Model:
    while True:
        clear_terminal()
        dis_select_model()
        second_error = False

        if adapter == 'attentionmodel': models = get_models(f'{MODEL_ROUTE}attentional/Sx{H}/')
        if adapter == 'linealmodel': models = get_models(f'{MODEL_ROUTE}lineal/{S}x{H}/')

        if models is None: 
            if adapter == 'attentionmodel': print(f'No hay modelos para seleccionar con altura {H}.')
            if adapter == 'linealmodel': print(f'No hay modelos para seleccionar con altura {H} y {S} stacks.')
            return None

        while True:
            num_model = input('Seleccione el modelo que desea utilizar: ')
            if num_model.isdigit() and 0 < int(num_model) <= len(models) + 1: 
                num_model = int(num_model)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero válido.')
            second_error = True

        if adapter == 'attentionmodel': model = load_cpmp_model(f'{MODEL_ROUTE}attentional/Sx{H}/{models[num_model]}')
        if adapter == 'linealmodel': model = load_cpmp_model(f'{MODEL_ROUTE}lineal{S}x{H}/{models[num_model]}')

        return model

def generate_v1_menu():
    while True:
        clear_terminal()
        dis_menu_gen1()
        second_error = False

        while True:
            S = input('Cuantos stacks contiene el problema? (S ≥ 1) ')
            if S.isdigit() and int(S) >= 1: 
                S = int(S)
                second_error = False
                break
    
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero.')
            second_error = True

        while True:
            H = input('De que altura es el problema? (H ≥ 3) ')
            if H.isdigit() and int(H) >= 3:
                H = int(H)
                second_error = False 
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero.')
            second_error = True

        max_N = S * (H - 2)
        while True:
            N = input(f'Cuantos contenedores posee el problema? (N ≥ 1 y N ≤ {max_N}) ')
            if N.isdigit() and int(N) <= max_N and int(N) >= 1: 
                N = int(N)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print(f'Por favor, coloque un número entero o menor a {max_N} y mayor a 1.')
            second_error = True

        while True:
            max_steps = input('En cuantos pasos como máximo se debe resolver el problema? (n ≥ 1) ')
            if max_steps.isdigit() and int(max_steps) >= 1: 
                max_steps = int(max_steps)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True:    
            perms_by_layout = input('Cuantas permutaciones necesita? (n ≥ 1) ')
            if perms_by_layout.isdigit() and int(perms_by_layout) >= 1: 
                perms_by_layout = int(perms_by_layout)
                second_error = False
                break
            
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True:
            sample_size = input('Cuantos datos necesita en total? (n ≥ 1) ')
            if sample_size.isdigit() and int(sample_size) >= 1: 
                sample_size = int(sample_size)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True: 
            optimizer = input('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ').lower()
            if optimizer in ['greedyv1', 'greedyv2', 'greedymodel']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)   
            else: delete_terminal_lines(2)
            print('Por favor, coloque un optimizador válido.')
            second_error = True
        
        while True:
            adapter_selected = input('Que adaptador necesitas? (AttentionModel, LinealModel) ').lower()
            if adapter_selected in ['attentionmodel', 'linealmodel']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un adaptador válido.')
            second_error = True
        
        while True:
            verbose = input('Desea ver la cantidad de datos generados durante la ejecución? (S / N) ').lower()
            if verbose in ['s', 'n']: 
                second_error = False
                break
            
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True
        
        while True:
            act = input('Está seguro de sus elecciones? (S / N) ').lower()
            if act in ['s', 'n']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True

        if act == 'n':
            clear_terminal()
            return
        
        if adapter_selected == 'attentionmodel': adapter = AttentionModel()
        elif adapter_selected == 'linealmodel': adapter = LinealModel()

        if optimizer == 'greedyv1': optimizer = GreedyV1()
        elif optimizer == 'greedyv2': optimizer = GreedyV2()
        elif optimizer == 'greedymodel': 
            model = select_model(S, H, adapter_selected)
            if model is None: 
                if try_again(): continue
                else: break

            optimizer = GreedyModel(model, adapter)

        if verbose == 's': verbose = True
        elif verbose == 'n': verbose = False

        print('La generación de datos comienza!\n')

        data, labels = generate_data_v1(S, H, N, sample_size, verbose= verbose, perms_by_layout= perms_by_layout, 
                                        solver= optimizer, adapter= adapter, max_steps= max_steps)
        
        print('Datos generados con éxito!')
        act = input('Desea guardar los datos generados? (S / N) ').lower()
        if act == 's':
            save_data_menu(data, labels)
        else:
            print('Datos no guardados.')

        print('')
        act = input('Desea volver a generar datos? (S / N) ').lower()
        if act == 'n':
            clear_terminal()
            break
        else: continue

def generate_v2_menu():
    while True:
        clear_terminal()
        dis_menu_gen2()
        second_error = False
        adapter = AttentionModel()

        while True:
            min_S = input('Cuantos stacks por lo mínimo contiene el problema? (S ≥ 1) ')
            if min_S.isdigit() and int(min_S) >= 1: 
                min_S = int(min_S)
                second_error = False
                break
    
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True:
            max_S = input(f'Cuantos stacks en máximo contiene el problema? (S ≥ {min_S}) ')
            if max_S.isdigit() and int(max_S) >= 1: 
                max_S = int(max_S)
                second_error = False
                break
    
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print(f'Por favor, coloque un número entero mayor a {min_S}.')
            second_error = True

        while True:
            H = input('De que altura es el problema? (H ≥ 3) ')
            if H.isdigit():
                H = int(H)
                second_error = False 
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero.')
            second_error = True

        while True:
            lb = input('Cual es el porcentaje mínimo de pasos que desea? (0 < lb ≤ 1) ').replace(',', '.')
            if lb.replace('.', '').isdigit() and 0 < float(lb) <= 1:
                lb = 1 - float(lb)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número decimal mayor a 0 y menor o igual a 1.')
            second_error = True

        while True:
            sample_size = input('Cuantos datos necesita en total? (n ≥ 1) ')
            if sample_size.isdigit() and int(sample_size) >= 1: 
                sample_size = int(sample_size)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True

        while True:
            batch_size = input('Cuantos datos desea generar por lote? (n ≥ 1 y el predeterminado 32) ')
            if batch_size.isdigit() and int(batch_size) >= 1:
                batch_size = int(batch_size)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True

        while True:
            optimizer = input('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ').lower()
            if optimizer in ['greedyv1', 'greedyv2', 'greedymodel']:
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un optimizador válido.')
            second_error = True
        
        while True:
            verbose = input('Desea ver la cantidad de datos generados durante la ejecución? (S / N) ').lower()
            if verbose in ['s', 'n']: 
                second_error = False
                break
            
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True
        
        while True:
            act = input('Está seguro de sus elecciones? (S / N) ').lower()
            if act in ['s', 'n']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True

        if act == 'n':
            clear_terminal()
            return

        if optimizer == 'greedyv1': optimizer = GreedyV1()
        elif optimizer == 'greedyv2': optimizer = GreedyV2()
        elif optimizer == 'greedymodel': 
            model = select_model(min_S, H, 'attentionmodel')
            if model is None: 
                if try_again(): continue
                else: break

            optimizer = GreedyModel(model, adapter)

        if verbose == 's': verbose = True
        elif verbose == 'n': verbose = False

        print('La generación de datos comienza!\n')

        input('Presione Enter para continuar...')
        data, labels = generate_data_v2(min_S, max_S, H, size= sample_size, lb= lb, verbose= verbose, 
                                        optimizer= optimizer, adapter= adapter, batch_size= batch_size)
        
        print('Datos generados con éxito!')
        act = input('Desea guardar los datos generados? (S / N) ').lower()
        if act == 's':
            save_data_menu(data, labels)
        else:
            print('Datos no guardados.')

        print('')
        act = input('Desea volver a generar datos? (S / N) ').lower()
        if act == 'n':
            clear_terminal()
            break
        else: continue

def generate_v3_menu():
    while True:
        clear_terminal()
        dis_menu_gen3()
        second_error = False

        while True:
            S = input('Cuantos stacks contiene el problema? (S ≥ 1) ')
            if S.isdigit() and int(S) >= 1: 
                S = int(S)
                second_error = False
                break
    
            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero.')
            second_error = True

        while True:
            H = input('De que altura es el problema? (H ≥ 3) ')
            if H.isdigit() and int(H) >= 3:
                H = int(H)
                second_error = False 
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero.')
            second_error = True

        max_N = S * (H - 2)
        while True:
            N = input(f'Cuantos contenedores posee el problema? (N ≥ 1 y N ≤ {max_N}) ')
            if N.isdigit() and int(N) <= max_N and int(N) >= 1: 
                N = int(N)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print(f'Por favor, coloque un número entero o menor a {max_N} y mayor a 1.')
            second_error = True

        while True:
            max_steps = input('En cuantos pasos como máximo se debe resolver el problema? (n ≥ 1) ')
            if max_steps.isdigit() and int(max_steps) >= 1: 
                max_steps = int(max_steps)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True:    
            perms_by_layout = input('Cuantas permutaciones necesita? (n ≥ 1) ')
            if perms_by_layout.isdigit() and int(perms_by_layout) >= 1: 
                perms_by_layout = int(perms_by_layout)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True:
            sample_size = input('Cuantos datos necesita en total? (n ≥ 1) ')
            if sample_size.isdigit() and int(sample_size) >= 1: 
                sample_size = int(sample_size)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True

        while True:
            bath_size = input('Cuantos datos desea generar por lote? (n ≥ 1 y el predeterminado 32) ')
            if bath_size.isdigit() and int(bath_size) >= 1:
                bath_size = int(bath_size)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero mayor a 1.')
            second_error = True
        
        while True:
            optimizer = input('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ').lower()
            if optimizer in ['greedyv1', 'greedyv2', 'greedymodel']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)   
            else: delete_terminal_lines(2)
            print('Por favor, coloque un optimizador válido.')
            second_error = True
        
        while True:
            selected_adapter = input('Que adaptador necesitas? (AttentionModel, LinealModel) ').lower()
            if selected_adapter in ['attentionmodel', 'linealmodel']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un adaptador válido.')
            second_error = True
        
        while True:
            verbose = input('Desea ver la cantidad de datos generados durante la ejecución? (S / N) ').lower()
            if verbose in ['s', 'n']:
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True
        
        while True:
            act = input('Está seguro de sus elecciones? (S / N) ').lower()
            if act in ['s', 'n']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True

        if act == 'n':
            clear_terminal()
            return
        
        if selected_adapter == 'attentionmodel': adapter = AttentionModel()
        elif selected_adapter == 'linealmodel': adapter = LinealModel()
        
        if optimizer == 'greedyv1': optimizer = GreedyV1()
        elif optimizer == 'greedyv2': optimizer = GreedyV2()
        elif optimizer == 'greedymodel': 
            model = select_model(S, H, selected_adapter)           
            if model is None: 
                if try_again(): continue
                else: break

            optimizer = GreedyModel(model, adapter)

        if verbose == 's': verbose = True
        elif verbose == 'n': verbose = False

        print('La generación de datos comienza!\n')

        data, labels = generate_data_v3(solver= optimizer, adapter= adapter, S= S, H= H, N= N, 
                                        sample_size= sample_size, batch_size= bath_size, 
                                        perms_by_layout= perms_by_layout, verbose= verbose, 
                                        max_steps= max_steps)

        print('Datos generados con éxito!')
        act = input('Desea guardar los datos generados? (S / N) ').lower()
        if act == 's':
            save_data_menu(data, labels)
        else:
            print('Datos no guardados.')

        print('')
        act = input('Desea volver a generar datos? (S / N) ').lower()
        if act == 'n':
            clear_terminal()
            break
        else: continue

def change_model_route_menu():
    global MODEL_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_model_route()
        second_error = False

        print(f'Actual ruta de los modelos: {MODEL_ROUTE}\n')

        model_route = input('Indique la nueva ruta de los modelos: ')
        if not os.path.exists(model_route): 
            print('La ruta no existe.')
            while True:
                act = input('Desea crear la ruta? (S / N) ').lower()
                if act in ['s', 'n']: 
                    second_error = False
                    break

                if not second_error: delete_terminal_lines(1)
                else: delete_terminal_lines(2)
                print('Por favor, coloque una opción válida.')
                second_error = True

            while True:
                act = input('Está seguro de la nueva ruta de los modelos? (S / N) ').lower()
                if act in ['s', 'n']: 
                    second_error = False
                    break

                if not second_error: delete_terminal_lines(1)
                else: delete_terminal_lines(2)
                print('Por favor, coloque una opción válida.')
                second_error = True

            if act == 's': 
                os.makedirs(model_route)
                print('Ruta creada con éxito!')
                break
            else: 
                print('Ruta no creada.')
                if try_again(): continue
                break
    input('Pulse Enter para continuar')
    MODEL_ROUTE = model_route

def change_uri_mongo_menu():
    global MONGO_URI
    global DB_HOST
    global DB_USER
    global DB_PASSWORD
    global DB_NAME

    while True:
        clear_terminal()
        dis_menu_change_mongodb_uri()
        second_error = False

        print(f'Actual datos de MongoDB:')
        print(f'1) HOST: {DB_HOST}')
        print(f'2) USER: {DB_USER}')
        print(f'3) PASSWORD: {DB_PASSWORD}\n')
        print(f'4) AUTENTICATION DATABASE: {DB_NAME}\n')

        while True:
            act = input('Desea cambiar los datos de MongoDB? (S / N) ').lower()
            if act in ['s', 'n']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True
        
        if act == 'n': break

        while True:
            opt = input('Que dato desea cambiar? (1 - 4) ')
            if opt.isdigit() and 0 < int(opt) <= 4:
                opt = int(opt)
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque un número entero válido.')
            second_error = True
        
        if opt == 1:
            DB_HOST = input('Indique la IP del servidor mongodb (IP:Port): ')
        elif opt == 2:
            DB_USER = input('Indique su nombre de usuario: ')
        elif opt == 3:
            while True:
                temp = getpass.getpass('Introduce tu contraseña: ')
                temp_1 = getpass.getpass('Confirme tu contraseña: ')
                
                if temp != temp_1:
                    print('Las contraseñas no coinciden.')
                    input('Pulse Enter para continuar')
                    
                    delete_terminal_lines(4)
                    continue

                DB_PASSWORD = temp
        elif opt == 4:
            DB_NAME = input('Ingrese el nombre de la base de datos de autenticación: ')

        MONGO_URI = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/?authSource={DB_NAME}'
        print('Datos cambiados con éxito!')

        while True:
            act = input('Desea cambiar otro dato? (S / N) ').lower()
            if act in ['s', 'n']: 
                second_error = False
                break

            if not second_error: delete_terminal_lines(1)
            else: delete_terminal_lines(2)
            print('Por favor, coloque una opción válida.')
            second_error = True

def change_data_route_menu():
    global JSON_DATA_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_data_route()
        second_error = False

        print(f'Actual ruta de los datos: {JSON_DATA_ROUTE}\n')

        data_route = input('Indique la nueva ruta de los datos: ')
        if not os.path.exists(data_route): 
            print('La ruta no existe.')
            while True:
                act = input('Desea crear la ruta? (S / N) ').lower()
                if act in ['s', 'n']: 
                    second_error = False
                    break

                if not second_error: delete_terminal_lines(1)
                else: delete_terminal_lines(2)
                print('Por favor, coloque una opción válida.')
                second_error = True

            while True:
                act = input('Está seguro de la nueva ruta de los datos? (S / N) ').lower()
                if act in ['s', 'n']: 
                    second_error = False
                    break

                if not second_error: delete_terminal_lines(1)
                else: delete_terminal_lines(2)
                print('Por favor, coloque una opción válida.')
                second_error = True

            if act == 's': 
                os.makedirs(data_route)
                print('Ruta creada con éxito!')
            else: 
                print('Ruta no creada.')
                input('Pulse Enter para continuar')
                if try_again(): continue
                break
    
    JSON_DATA_ROUTE = data_route

def option_menu():
    while True:
        clear_terminal()
        dis_option_menu()

        try:
            option = int(input('Escoja una opción: '))
        except ValueError:
            print('Coloque una opción valida (1 - 4)')
            input('Pulse Enter para continuar')

            clear_terminal()
            continue
        
        if option == 1:
            change_model_route_menu()
        elif option == 2:
            change_uri_mongo_menu()
        elif option == 3:
            change_data_route_menu()
        elif option == 4:
            break
        else:
            print('opción invalida, por favor colocar un valor entre el 1 al 4')
            input('Pulse Enter para continuar')

        clear_terminal()

def save_program_data():
    global DB_HOST
    global DB_USER
    global DB_PASSWORD
    global DB_NAME
    global MODEL_ROUTE
    global JSON_DATA_ROUTE

    file = open('.env', 'w', encoding='utf-8')

    file.write(f'# Datos de conexión a la base de datos\n')
    file.write(f'DB_HOST="{DB_HOST}"\n')
    file.write(f'DB_USER="{DB_USER}"\n')
    file.write(f'DB_PASSWORD="{DB_PASSWORD}"\n')
    file.write(f'DB_NAME="{DB_NAME}"\n')
    file.write('\n')
    file.write(f'# Ubicación de archivos\n')
    file.write(f'MODEL_ROUTE="{MODEL_ROUTE}"\n')
    file.write(f'JSON_DATA_ROUTE="{JSON_DATA_ROUTE}"\n')

    file.close()

def main_menu():
    try:
        while True:
            clear_terminal()
            dis_main_menu()
            
            try:
                option = int(input('Escoja una opción: '))
            except ValueError:
                print('Coloque una opción valida (1 - 6)')
                input('Pulse Enter para continuar')

                clear_terminal()
                continue
            
            if option == 1:
                generate_v1_menu()
            elif option == 2:
                generate_v2_menu()
            elif option == 3:
                generate_v3_menu()
            elif option == 4:
                option_menu()
            elif option == 5:
                print('')
                print('Cerrando el programa...')

                save_program_data()
                exit()
            else:
                print('opción invalida, por favor colocar un valor entre el 1 al 6')
                input('Pulse Enter para continuar')

            clear_terminal()
    except KeyboardInterrupt:
            print('')
            print('Cerrando el programa...')

            save_program_data()
            exit()

if __name__ == '__main__':
    main_menu()