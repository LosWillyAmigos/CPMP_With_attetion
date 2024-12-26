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
from urllib.parse import quote_plus
from dotenv import load_dotenv
from typing import Callable
import getpass
import sys

SYSTEM = os.name

def delete_terminal_lines(lines: int) -> None:
    for _ in range(lines):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

def clear_terminal() -> None:
    if SYSTEM == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def input_label(input_text: str, error_text: str, verify_input: Callable[[str], bool]) -> str:
    input_jump_line_count = input_text.count('\n') + 1
    error_jump_line_count = error_text.count('\n') + 1
    second_error = False

    while True:
        text = input(input_text)
        if verify_input(text):
            break
        
        if not second_error: delete_terminal_lines(input_jump_line_count)
        else: delete_terminal_lines(error_jump_line_count + input_jump_line_count)
        print(error_text)
        second_error = True

    return text

def input_password() -> str:
    while True:
        temp = getpass.getpass('Introduce tu contraseña: ')
        temp_1 = getpass.getpass('Confirme tu contraseña: ')
        
        if temp != temp_1:
            print('Las contraseñas no coinciden.')
            input('Pulse Enter para continuar')
            
            delete_terminal_lines(4)
            continue
            
        break

    return temp

def write_environment(db_host: str, 
                      db_user: str, 
                      db_password: str, 
                      db_name: str, 
                      model_route: str, 
                      json_data_route: str) -> None:
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

def install_environment() -> tuple:
    db_host = ''
    db_user = ''
    db_password = ''
    db_name = ''
    model_route = os.getcwd().replace('\\', '/') + '/models/'
    json_data_route = os.getcwd().replace('\\', '/') + '/data/'

    while True:
        clear_terminal()
    
        print('Bienvenido al instalador de CPMP_With_Attention\n')
        opt = input_label('Usted usa MongoDB? (S/N) ', 
                        'Por favor, coloque una opción válida.', 
                        lambda x: x.lower() in ['s', 'n']).lower()
        if opt == 's':    
            db_host = input('Indique la IP del servidor mongodb (IP:Port): ')
            db_user = input('Indique su nombre de usuario: ')
            db_password = getpass.getpass('Introduce tu contraseña: ')
            db_name = input('Ingrese el nombre de la base de datos de autenticación: ')
            db_uri = f'mongodb://{db_user}:{"*" * (len(db_password) + 1)}@{db_host}/?authSource={db_name}'
            
            print('MongoDB URI:', db_uri)
            opt = input_label('Está seguro de los datos ingresados? (S/N) ', 
                            'Por favor, coloque una opción válida.', 
                            lambda x: x.lower() in ['s', 'n']).lower()

            if opt.lower() == 'n': continue
        
        print(f'La ruta de los modelos será: {model_route}')
        opt = input_label('Está seguro de la ruta de los modelos? (S/N) ', 
                        'Por favor, coloque una opción válida.', 
                        lambda x: x.lower() in ['s', 'n']).lower()

        if opt == 'n': model_route = input('Indique la ruta de los modelos: ')
        
        print(f'La ruta de los datos generados por JSON será: {json_data_route}')
        opt = input_label('Está seguro de la ruta de los datos generados por JSON? (S/N) ', 
                        'Por favor, coloque una opción válida.', 
                        lambda x: x.lower() in ['s', 'n']).lower()
        
        if opt == 'n': json_data_route = input('Indique la ruta de los datos generados por JSON: ')

        write_environment(db_host, db_user, db_password, db_name, model_route, json_data_route)

        return json_data_route, model_route

def install_data_route(route) -> None:
    os.makedirs(f'{route}attentional/')
    os.makedirs(f'{route}lineal/')

def install_program() -> None:
    json_route, model_route = install_environment()
    if not os.path.exists(model_route): install_data_route(MODEL_ROUTE)
    if not os.path.exists(json_route): install_data_route(JSON_DATA_ROUTE)

if not os.path.exists('.env'): install_program()

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

MONGO_URI = f'mongodb://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}/?authSource={DB_NAME}'
MODEL_ROUTE = os.environ.get("MODEL_ROUTE")
JSON_DATA_ROUTE = os.environ.get("JSON_DATA_ROUTE")

def try_again() -> bool:
    act = input('Quiere volver a intentar? (S / N) ').lower()

    delete_terminal_lines(1)

    if act == 's': return True
    return False

def dis_menu_gen1() -> None:
    print('|*|*************| Generador V1 |*************|*|')
    print('|*| Los datos que entregará este generador   |*|')
    print('|*| será el estado inicial de un problema    |*|')
    print('|*| con su solución.                         |*|')
    print('|*|**********| CPMP_With_Attention |*********|*|')
    print('')

def dis_menu_gen2() -> None:
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

def dis_menu_gen3() -> None:
    print('|*|*************| Generador V3 |*************|*|')
    print('|*| Los datos que entregará este generador   |*|')
    print('|*| será el estado incial de un problema     |*|')
    print('|*| con su solución, pero ampliando su       |*|')
    print('|*| búsqueda hasta un primer nivel de        |*|')
    print('|*| profundidad.                             |*|')
    print('|*|**********| CPMP_With_Attention |*********|*|')
    print('') 

def dis_menu_change_model_route() -> None:
    print('|*|*************| Ruta de los modelos |*************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los modelos.                |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_menu_change_mongodb_uri() -> None:
    print('|*|*************| URI de MongoDB |***************|*|')
    print('|*| Este menú le dará apoyo para cambiar la URI  |*|')
    print('|*| de conexión a MongoDB.                       |*|')
    print('|*|***********| CPMP_With_Attention |************|*|')
    print('')

def dis_menu_change_data_route() -> None:
    print('|*|**************| Ruta de los datos |**************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los datos generados.        |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_save_data_json() -> None:
    print('|*|*************| JSON |**************|*|')
    print('|*| Este menú le dará apoyo para      |*|')
    print('|*| almacenar los datos generados en  |*|')
    print('|*| formato JSON dentro de su unidad  |*|')
    print('|*| de almacenamiento local.          |*|')
    print('|*|******| CPMP_With_Attention |******|*|')
    print('')

def dis_save_data_mongodb() -> None:
    print('|*|*************| MongoDB |*************|*|')
    print('|*| Este menú le dará apoyo para        |*|')
    print('|*| almacenar los datos generados en    |*|')
    print('|*| MongoDB.                            |*|')
    print('|*|*******| CPMP_With_Attention |*******|*|')
    print('')

def dis_select_model() -> None:
    print('|*|***************| Selección de modelo |***************|*|')
    print('|*| Este menú le dará apoyo para seleccionar el modelo  |*|')
    print('|*| que desea utilizar para la generación de datos.     |*|')
    print('|*|***************| CPMP_With_Attention |***************|*|')
    print('')

def dis_save_data() -> None:
    print('|*|*************| Guardado de datos |*************|*|')
    print('|*| Como desea almacenar sus datos?               |*|')
    print('|*| 1) Formato JSON                               |*|')
    print('|*| 2) Cluster MongoDB                            |*|')
    print('|*|************| CPMP_With_Attention |************|*|')
    print('')

def dis_option_menu() -> None:
    print('|*|**************| Opciones |****************|*|')
    print('|*| 1) Cambiar la ruta de los modelos        |*|')
    print('|*| 2) Cambiar la URI de MongoDB             |*|')
    print('|*| 3) Cambiar la ubicación de los archivos  |*|')
    print('|*| 4) Volver al menú principal              |*|')
    print('|*|*********| CPMP_With_Attention |**********|*|')
    print('')

def dis_main_menu() -> None:
    print('|*|*************| Menú del generador |*************|*|')
    print('|*|  1) Generar datos V1                           |*|')
    print('|*|  2) Generar datos V2 (Solo para Atención)      |*|')
    print('|*|  3) Generar datos V3                           |*|')
    print('|*|  4) Opciones                                   |*|')
    print('|*|  5) Salir                                      |*|')
    print('|*|************| CPMP_With_Attention |*************|*|')
    print('')

def change_uri_mongo_menu() -> None:
    global MONGO_URI
    global DB_HOST
    global DB_USER
    global DB_PASSWORD
    global DB_NAME

    while True:
        clear_terminal()
        dis_menu_change_mongodb_uri()

        print(f'Actual datos de MongoDB:')
        print(f'1) HOST: {DB_HOST}')
        print(f'2) USER: {DB_USER}')
        print(f"3) PASSWORD: {'*' * len(DB_PASSWORD)}")
        print(f'4) AUTENTICATION DATABASE: {DB_NAME}\n')

        act = input_label('Desea cambiar los datos de MongoDB? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n': break

        opt = int(input_label('Que dato desea cambiar? (1 - 4) ', 
                          'Por favor, coloque un número entero.', 
                          lambda x: x.isdigit() and 0 < int(x) <= 4))
        
        if opt == 1:
            DB_HOST = input('Indique la IP del servidor mongodb (IP:Port): ')
        elif opt == 2:
            DB_USER = input('Indique su nombre de usuario: ')
        elif opt == 3:
            DB_PASSWORD = input_password()
        elif opt == 4:
            DB_NAME = input('Ingrese el nombre de la base de datos de autenticación: ')

        MONGO_URI = f'mongodb://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}/?authSource={DB_NAME}'
        print('Datos cambiados con éxito!')

        act = input_label('Desea cambiar otro dato? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 'n': break

def change_data_route_menu() -> None:
    global JSON_DATA_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_data_route()

        print(f'Actual ruta de los datos: {JSON_DATA_ROUTE}\n')

        data_route = input('Indique la nueva ruta de los datos (puede ser la misma): ')
        if not os.path.exists(data_route): print('La ruta no existe.')
        else: print('Ruta ya existe.')

        act = input_label('Está seguro de la nueva ruta de los datos? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()

        if act == 's': 
            install_data_route(data_route)
            print('Ruta creada con éxito!')
        else: 
            print('Ruta no creada.')
            input('Pulse Enter para continuar')
            if try_again(): continue
        
        break

    input('Pulse Enter para continuar')
    JSON_DATA_ROUTE = data_route

def change_model_route_menu() -> None:
    global MODEL_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_model_route()

        print(f'Actual ruta de los modelos: {MODEL_ROUTE}\n')
        
        model_route = input('Indique la nueva ruta de los modelos (puede ser la misma): ').replace('\\', '/')
        if not model_route.endswith('/'): model_route += '/'
        if not os.path.exists(model_route): print('La ruta no existe.')
        else: print('Ruta ya existe.')

        act = input_label('Está seguro de la nueva ruta de los modelos? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()

        if act == 's': 
            install_data_route(model_route)
            print('Ruta creada con éxito!')
            break
        else: 
            print('Ruta no creada.')
            if try_again(): continue
            break
    input('Pulse Enter para continuar')
    MODEL_ROUTE = model_route

def save_data_mongo_menu(data: list, labels: list) -> bool:
    while True:
        clear_terminal()
        dis_save_data_mongodb()

        client = connect_to_server(MONGO_URI)
        if client is None and try_again():
            print('La conexión no se pudo realizar.')
            opt = input_label('Desea cambiar la URI de MongoDB? (S / N) ',
                                'Por favor, coloque una opción válida.', 
                                lambda x: x.lower() in ['s', 'n']).lower()
            if opt == 's': 
                change_uri_mongo_menu()
                continue

            return False
        
        data_base_name = input_label('Indique el nombre de la base de datos: ', 
                                     'Por favor, coloque un nombre válido.', 
                                     lambda x: x in client.list_database_names())
        
        collection_name = input_label('Indique el nombre de la colección: ', 
                                      'La colección no existe.', 
                                      lambda x: x in client[data_base_name].list_collection_names())

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

def save_data_json_menu(data: list, labels: list) -> bool:
    while True:
        clear_terminal()
        dis_save_data_json()

        if not os.path.exists(JSON_DATA_ROUTE):
            print('La ruta de los datos no existe.')
            opt = input_label('Desea cambiar la ruta? (S / N) ', 
                              'Por favor, coloque una opción válida.', 
                              lambda x: x.lower() in ['s', 'n']).lower()
            if opt == 's': 
                change_data_route_menu()
                continue

            input('Pulse Enter para continuar')
            
            return False

        file_name = input('Indique el nombre del archivo: ')
        if os.path.exists(JSON_DATA_ROUTE + file_name + '.json'): 
            print('El archivo ya existe.')
            print('Si continua, el archivo será sobreescrito.')

            option = input_label('Desea continuar? (S / N) ', 
                                 'Por favor, coloque una opción válida.', 
                                 lambda x: x.lower() in ['s', 'n']).lower()
            
            if option == 'n': return False

        print('\nIniciando Guardado...')
        try: 
            save_data_json(data, labels, JSON_DATA_ROUTE + file_name)
        except Exception as e:
            print('Error al guardar los datos.\n')
            print(e)	
            input('Pulse Enter para continuar')

            return False

        print('Datos guardados con éxito!')
        return True

def save_data_menu(data: list, labels: list) -> None:
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

        input('Pulse Enter para continuar')
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
        
        if not os.path.exists(MODEL_ROUTE):
            print('La ruta de los modelos no existe.')
            opt = input_label('Desea cambiar la ruta de los modelos? (S / N) ', 
                              'Por favor, coloque una opción válida.', 
                              lambda x: x.lower() in ['s', 'n']).lower()
            if opt == 's': 
                change_model_route_menu()
                continue

            return None

        if adapter == 'attentionmodel' and os.path.exists(f'{MODEL_ROUTE}attentional/Sx{H}/'): models = get_models(f'{MODEL_ROUTE}attentional/Sx{H}/')
        elif adapter == 'attentionmodel' and not os.path.exists(f'{MODEL_ROUTE}attentional/Sx{H}/'): 
            os.mkdir(f'{MODEL_ROUTE}attentional/Sx{H}/')
            models = get_models(f'{MODEL_ROUTE}attentional/Sx{H}/')

        if adapter == 'linealmodel' and os.path.exists(f'{MODEL_ROUTE}lineal/{S}x{H}/'): models = get_models(f'{MODEL_ROUTE}lineal/{S}x{H}/')
        elif adapter == 'linealmodel' and not os.path.exists(f'{MODEL_ROUTE}lineal/{S}x{H}/'): 
            os.mkdir(f'{MODEL_ROUTE}lineal/{S}x{H}/')
            models = get_models(f'{MODEL_ROUTE}lineal/{S}x{H}/')

        if models is None: 
            if adapter == 'attentionmodel': print(f'No hay modelos para seleccionar con altura {H}.')
            if adapter == 'linealmodel': print(f'No hay modelos para seleccionar con altura {H} y {S} stacks.')
            return None

        num_model = int(input_label('Seleccione el modelo que desea utilizar: ', 
                                'Por favor, coloque un número entero.', 
                                lambda x: x.isdigit() and 0 < int(x) <= len(models) + 1))

        if adapter == 'attentionmodel': model = load_cpmp_model(f'{MODEL_ROUTE}attentional/Sx{H}/{models[num_model]}')
        if adapter == 'linealmodel': model = load_cpmp_model(f'{MODEL_ROUTE}lineal{S}x{H}/{models[num_model]}')

        return model
    

def generate_v1_menu() -> None:
    while True:
        clear_terminal()
        dis_menu_gen1()

        S = int(input_label('Cuantos stacks contiene el problema? (S ≥ 1) ', 
                            'Por favor, coloque un número entero.', 
                            lambda x: x.isdigit() and int(x) >= 1))

        H = int(input_label('De que altura es el problema? (H ≥ 3) ', 
                            'Por favor, coloque un número entero.', 
                            lambda x: x.isdigit() and int(x) >= 3))

        max_N = S * (H - 2)
        N = int(input_label(f'Cuantos contenedores posee el problema? (N ≥ 1 y N ≤ {max_N}) ', 
                            f'Por favor, coloque un número entero o menor a {max_N} y mayor a 1.', 
                            lambda x: x.isdigit() and int(x) <= max_N and int(x) >= 1))

        max_steps = int(input_label('En cuantos pasos como máximo se debe resolver el problema? (n ≥ 1) ', 
                                    'Por favor, coloque un número entero mayor a 1.', 
                                    lambda x: x.isdigit() and int(x) >= 1))
        
        perms_by_layout = int(input_label('Cuantas permutaciones necesita? (n ≥ 1) ', 
                                         'Por favor, coloque un número entero mayor a 1.', 
                                         lambda x: x.isdigit() and int(x) >= 1))
        
        sample_size = int(input_label('Cuantos datos necesita en total? (n ≥ 1) ', 
                                     'Por favor, coloque un número entero mayor a 1.', 
                                     lambda x: x.isdigit() and int(x) >= 1))
        
        optimizer = input_label('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ', 
                                'Por favor, coloque un optimizador válido.', 
                                lambda x: x.lower() in ['greedyv1', 'greedyv2', 'greedymodel']).lower()
        
        adapter_selected = input_label('Que adaptador necesitas? (AttentionModel, LinealModel) ', 
                                      'Por favor, coloque un adaptador válido.', 
                                      lambda x: x.lower() in ['attentionmodel', 'linealmodel']).lower()
        
        verbose = input_label('Desea ver la cantidad de datos generados durante la ejecución? (S / N) ', 
                             'Por favor, coloque una opción válida.', 
                             lambda x: x.lower() in ['s', 'n']).lower()
        
        act = input_label('Está seguro de sus elecciones? (S / N) ', 
                         'Por favor, coloque una opción válida.', 
                         lambda x: x.lower() in ['s', 'n']).lower()

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

        print('\nLa generación de datos comienza!')
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

def generate_v2_menu() -> None:
    while True:
        clear_terminal()
        dis_menu_gen2()
        adapter = AttentionModel()

        min_S = int(input_label('Cuantos stacks por lo mínimo contiene el problema? (S ≥ 1) ', 
                                'Por favor, coloque un número entero mayor a 1.', 
                                lambda x: x.isdigit() and int(x) >= 1))

        max_S = int(input_label(f'Cuantos stacks en máximo contiene el problema? (S ≥ {min_S}) ', 
                                f'Por favor, coloque un número entero mayor a {min_S}.', 
                                lambda x: x.isdigit() and int(x) >= 1))
        
        H = int(input_label('De que altura es el problema? (H ≥ 3) ', 
                            'Por favor, coloque un número entero.', 
                            lambda x: x.isdigit() and int(x) >= 3))
        
        lb = 1 - float(input_label('Cual es el porcentaje mínimo de pasos que desea? (0 < n ≤ 1) ', 
                                   'Por favor, coloque un número decimal mayor a 0 y menor o igual a 1.', 
                                   lambda x: x.replace('.', '').isdigit() and 0 < float(x) <= 1))
        
        sample_size = int(input_label('Cuantos datos necesita en total? (n ≥ 1) ', 
                                     'Por favor, coloque un número entero mayor a 1.', 
                                     lambda x: x.isdigit() and int(x) >= 1))
        
        batch_size = int(input_label('Cuantos datos desea generar por lote? (n ≥ 1 y el predeterminado 32) ', 
                                    'Por favor, coloque un número entero mayor a 1.', 
                                    lambda x: x.isdigit() and int(x) >= 1))
        
        num_threads = int(input_label(f'Cuantos hilos desea utilizar? (n ≥ 1 y n ≤ {os.cpu_count()}) ', 
                                    f'Por favor, coloque un número entero mayor a 1.', 
                                    lambda x: x.isdigit() and int(x) >= 1 and int(x) <= os.cpu_count()))
        
        optimizer = input_label('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ', 
                                'Por favor, coloque un optimizador válido.', 
                                lambda x: x.lower() in ['greedyv1', 'greedyv2', 'greedymodel']).lower()
        
        verbose = input_label('Desea ver la cantidad de datos generados durante la ejecución? (S / N) ', 
                             'Por favor, coloque una opción válida.', 
                             lambda x: x.lower() in ['s', 'n']).lower()
        
        act = input_label('Está seguro de sus elecciones? (S / N) ', 
                         'Por favor, coloque una opción válida.', 
                         lambda x: x.lower() in ['s', 'n']).lower() 
        
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

        print('\nLa generación de datos comienza!')
        data, labels = generate_data_v2(min_S, max_S, H, size= sample_size, lb= lb, verbose= verbose, 
                                        optimizer= optimizer, adapter= adapter, batch_size= batch_size, 
                                        num_threads= num_threads)
        
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

def generate_v3_menu() -> None:
    while True:
        clear_terminal()
        dis_menu_gen3()

        S = int(input_label('Cuantos stacks contiene el problema? (S ≥ 1) ', 
                            'Por favor, coloque un número entero.', 
                            lambda x: x.isdigit() and int(x) >= 1))
        
        H = int(input_label('De que altura es el problema? (H ≥ 3) ', 
                            'Por favor, coloque un número entero.', 
                            lambda x: x.isdigit() and int(x) >= 3))
        
        max_N = S * (H - 2)
        N = int(input_label(f'Cuantos contenedores posee el problema? (N ≥ 1 y N ≤ {max_N}) ', 
                            f'Por favor, coloque un número entero o menor a {max_N} y mayor a 1.', 
                            lambda x: x.isdigit() and int(x) <= max_N and int(x) >= 1))
        
        max_steps = int(input_label('En cuantos pasos como máximo se debe resolver el problema? (n ≥ 1) ', 
                                    'Por favor, coloque un número entero mayor a 1.', 
                                    lambda x: x.isdigit() and int(x) >= 1))
        
        perms_by_layout = int(input_label('Cuantas permutaciones necesita? (n ≥ 1) ', 
                                         'Por favor, coloque un número entero mayor a 1.', 
                                         lambda x: x.isdigit() and int(x) >= 1))
        
        sample_size = int(input_label('Cuantos datos necesita en total? (n ≥ 1) ', 
                                     'Por favor, coloque un número entero mayor a 1.', 
                                     lambda x: x.isdigit() and int(x) >= 1))

        bath_size = int(input_label('Cuantos datos desea generar por lote? (n ≥ 1 y el predeterminado 32) ', 
                                  'Por favor, coloque un número entero mayor a 1.', 
                                  lambda x: x.isdigit() and int(x) >= 1))
        
        optimizer = input_label('Que optimizador necesitas? (GreedyV1, GreedyV2, GreedyModel) ', 
                                'Por favor, coloque un optimizador válido.', 
                                lambda x: x.lower() in ['greedyv1', 'greedyv2', 'greedymodel']).lower()
        
        selected_adapter = input_label('Que adaptador necesitas? (AttentionModel, LinealModel) ', 
                                      'Por favor, coloque un adaptador válido.', 
                                      lambda x: x.lower() in ['attentionmodel', 'linealmodel']).lower()
        
        verbose = input_label('Desea ver la cantidad de datos generados durante la ejecución? (S / N) ', 
                             'Por favor, coloque una opción válida.', 
                             lambda x: x.lower() in ['s', 'n']).lower()
        
        act = input_label('Está seguro de sus elecciones? (S / N) ', 
                         'Por favor, coloque una opción válida.', 
                         lambda x: x.lower() in ['s', 'n']).lower()

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

def option_menu() -> None:
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

def save_program_data() -> None:
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

def main_menu() -> None:
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
    except (KeyboardInterrupt, EOFError) as e:
            print('')
            print('Cerrando el programa...')

            save_program_data()
            exit()

if __name__ == '__main__':
    main_menu()