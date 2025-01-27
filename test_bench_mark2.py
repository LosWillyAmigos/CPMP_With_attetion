import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from attentional_cpmp.model import load_cpmp_model
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.utils.adapters import LinealModel
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.optimizer import GreedyV1
from cpmp_ml.optimizer import GreedyV2
from keras.src.models import Model
from cpmp_ml.utils import Layout
from dotenv import load_dotenv
from typing import Callable
import pandas as pd
import numpy as np
import sys

SYSTEM = os.name

def delete_terminal_lines(lines: int):
    for _ in range(lines):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    sys.stdout.flush()

def clear_terminal() -> None:
    if SYSTEM == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def try_again() -> bool:
    act = input('Quiere volver a intentar? (S / N) ').lower()

    delete_terminal_lines(1)

    if act == 's': return True
    return False

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

def install_data_route(route) -> None:
    os.makedirs(f'{route}attentional/')
    os.makedirs(f'{route}lineal/')

load_dotenv()

MODEL_ROUTE = os.environ.get("MODEL_ROUTE")
BENCHMARK_ROUTE = os.environ.get("BENCHMARK_ROUTE")

def dis_main_menu() -> None:
    print('|*|************ | Menú de benchmark | ************|*|')
    print('|*| 1) Pruebas con modelo almacenado              |*|')
    print('|*| 2) Pruebas con modelo desde cero              |*|')
    print('|*| 3) Visualizar pruebas                         |*|')
    print('|*| 4) Opciones                                   |*|')
    print('|*| 5) Salir                                      |*|')
    print('|*|*********** | CPMP_With_Attention | ***********|*|')
    print('')

def dis_saved_model_test() -> None:
    print('|*|******* | Pruebas con modelo almacenado | *******|*|')
    print('|*| Este menú le dará apoyo para realizar pruebas   |*|')
    print('|*| con un modelo que tenga almacenado.             |*|')
    print('|*|************ | CPMP_With_Attention | ************|*|')
    print('')

def dis_new_model_test() -> None:
    print('|*|******* | Pruebas con modelo desde cero | *******|*|')
    print('|*| Este menú le dará apoyo para realizar pruebas   |*|')
    print('|*| con un modelo que se creará desde cero.         |*|')
    print('|*|************ | CPMP_With_Attention | ************|*|')
    print('')

def dis_view_tests() -> None:
    print('|*|************* | Visualizar pruebas | ************|*|')
    print('|*| Este menú le permitirá visualizar los           |*|')
    print('|*| resultados de las pruebas realizadas a través   |*|')
    print('|*| de una tabla mostrada por terminal.             |*|')
    print('|*|************ | CPMP_With_Attention | ************|*|')

def dis_options() -> None:
    print('|*|*************** | Opciones | *****************|*|')
    print('|*| 1) Cambiar ruta de los casos de prueba       |*|')
    print('|*| 2) Cambiar ruta de los modelos               |*|')
    print('|*| 3) Cambiar ruta de los resultados            |*|')
    print('|*| 4) Volver al menú principal                  |*|')
    print('|*|********** | CPMP_With_Attention | ***********|*|')
    print('')

def dis_menu_change_model_route() -> None:
    print('|*|*************| Ruta de los modelos |*************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta    |*|')
    print('|*| donde se encuentran los modelos.                |*|')
    print('|*|*************| CPMP_With_Attention |*************|*|')
    print('')

def dis_menu_change_benchmark_route() -> None:
    print('|*|*************| Ruta de los casos de prueba |*************|*|')
    print('|*| Este menú le dará apoyo para cambiar la ruta donde se   |*|')
    print('|*| encuentran los casos de prueba.                         |*|')
    print('|*|*****************| CPMP_With_Attention |*****************|*|')
    print('')

def dis_select_model() -> None:
    print('|*|***************| Selección de modelo |***************|*|')
    print('|*| Este menú le dará apoyo para seleccionar el modelo  |*|')
    print('|*| que desea utilizar para la generación de datos.     |*|')
    print('|*|***************| CPMP_With_Attention |***************|*|')
    print('')

def dis_show_directories() -> None:
    print('|*|********************| Directorios |*********************|*|')
    print('|*| Este menú le permitirá visualizar los directorios      |*|')
    print('|*| de los casos de prueba almacenados, escoja el a través |*|')
    print('|*| de un número los casos de prueba que desee utilizar.   |*|')
    print('|*|****************| CPMP_With_Attention |*****************|*|')
    print('')

def read_benchmark_file(route: str, H: int | None = None) -> tuple[Layout, int, int, int]:
    with open(route) as file:
        S, N = (int(x) for x in next(file).split())
        if H is None: H = (N / S) + 2

        stacks = []
        for line in file:
            stack = [int(x) for x in line.split()[1::]]
            stacks.append(stack)
        
        lay = Layout(stacks, H)

    return lay, S, H, N

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

def change_benchmark_route() -> None:
    global BENCHMARK_ROUTE

    while True:
        clear_terminal()
        dis_menu_change_benchmark_route()

        print(f'Actual ruta de los casos de prueba: {BENCHMARK_ROUTE}\n')

        benchmark_path = input('Indique la nueva ruta de los casos de prueba (puede ser la misma): ').replace('\\', '/')
        if not benchmark_path.endswith('/'): benchmark_path += '/'
        if not os.path.exists(benchmark_path): print('La ruta no existe.')
        else: print('Ruta ya existe.')

        act = input_label('Está seguro de la nueva ruta de los casos de prueba? (S / N) ', 
                          'Por favor, coloque una opción válida.', 
                          lambda x: x.lower() in ['s', 'n']).lower()
        
        if act == 's':
            os.mkdir(benchmark_path)
            break
        else:
            print('Ruta no creada.')
            if try_again(): continue
            break

    input('Pulse Enter para continuar')
    BENCHMARK_ROUTE = benchmark_path

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
    
def show_benchmarks_directories() -> str:
    benchmarks = dict()
    temp = dict()
    benchmarks_direct = os.listdir(BENCHMARK_ROUTE)

    if len(benchmarks_direct) == 0: return None

    for i, direct in enumerate(benchmarks_direct):
        temp.update({i + 1: direct})
        print(f'{i + 1}) {direct}')
    print('')

    test_type = int(input_label('Ingrese el tipo de casos de prueba: ', 
                                'Por favor, coloque un número entero.', 
                                lambda x: x.isdigit() and int(x) > 0 and int(x) <= len(os.listdir(BENCHMARK_ROUTE))))
    
    for i, direct in enumerate(os.listdir(f'{BENCHMARK_ROUTE}{temp[int(test_type)]}')):
        benchmarks.update({i + 1: direct})
        print(f'{i + 1}) {direct}')

    if benchmarks == {}: return None

    opt = int(input_label('Seleccione el directorio de prueba que desea utilizar: ',
                          'Por favor, coloque un número entero.', 
                          lambda x: x.isdigit() and int(x) > 0 and int(x) <= len(os.listdir(f'{BENCHMARK_ROUTE}{temp[int(test_type)]}'))))

    
    return temp[test_type] + '/' + benchmarks[opt]

def read_optimal_solution(route: str) -> int | pd.DataFrame:
    if route.endswith('.bay'):
        with open(route, 'r') as file:
            optimal = int(file.readline())
    elif route.endswith('.xlsx'):
        optimal = pd.read_excel(route, usecols='A:B')
        
    return optimal

def load_problems(path: str, total_problems: int, H: int | None = None) -> tuple[list[Layout], int, int, int, int | pd.DataFrame]:
    problems = []
    cont = 0

    for file in os.scandir(path):
        if cont == total_problems: break
        if file.is_file() and (file.name.endswith('.txt') or file.name.endswith('.xlsx')):
            optimal = read_optimal_solution(file.path)
        if file.is_file() and (file.name.endswith('.dat') or file.name.endswith('.bay')):
            lay, S, H, N = read_benchmark_file(file.path, H)
            problems.append(lay)

            cont += 1

    return problems, S, H, N, optimal

def run_experiments(
        problems: list[Layout], 
        problems_name: str, 
        S: int, 
        H: int, 
        N: int, 
        optimal: int | pd.DataFrame,
        model: Model,
        adapter: DataAdapter
) -> None:
    greedy1 = GreedyV1()
    greedy2 = GreedyV2()
    greedy_model = GreedyModel(model, adapter)

    cost_g1 = greedy1.solve(np.array(problems))[0]
    cost_g2 = greedy2.solve(np.array(problems), max_steps= N * 2)[0]
    cost_gmodel = greedy_model.solve(np.array(problems), max_steps= N * 2)[0]

    print(f'Costo Greedy 1: {cost_g1}')
    print(f'Costo Greedy 2: {cost_g2}')
    print(f'Costo Greedy Model: {cost_gmodel}')
    pass

def saved_model_test():
    while True:
        clear_terminal()
        dis_saved_model_test()

        dis_show_directories()
        directory = show_benchmarks_directories()
        if directory is None: 
            print('No hay casos de prueba para seleccionar.')
            opt = input_label('Desea cambiar la ruta de los casos de prueba? (S / N) ',
                              'Por favor, coloque una opción válida.',
                              lambda x: x.lower() in ['s', 'n']).lower()
            
            if opt == 's':
                change_benchmark_route()
                continue
            else: break
        
        total_cases = len(os.listdir(f'{BENCHMARK_ROUTE}{directory}')) - 1
        
        size_problems = int(input_label(f'Ingrese la cantidad de problemas que desea probar (1 - {total_cases}): ',
                                        f'Por favor, coloque un número entero entre el 1 y el {total_cases}.',
                                        lambda x: x.isdigit() and 1 <= int(x) <= total_cases))
        
        selected_adapter = input_label('Que adaptador necesitas? (AttentionModel, LinealModel) ', 
                                      'Por favor, coloque un adaptador válido.', 
                                      lambda x: x.lower() in ['attentionmodel', 'linealmodel']).lower()

        verbose = input_label('Desea ver la cantidad de datos predichos durante la ejecución? (S / N) ', 
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

        problems, S, H, N, optimal = load_problems(f'{BENCHMARK_ROUTE}{directory}/', size_problems)
        
        model = select_model(S, H, selected_adapter)
        if model is None: 
            if try_again(): continue
            else: break

        run_experiments(problems, directory, S, H, N, optimal, model, adapter)
          
def main_menu():
    try:
        while True:
            clear_terminal()
            dis_main_menu()
    
            option = int(input_label('Ingrese una opción: ', 
                                     'Coloque una opción valida (1 - 5)',
                                      lambda x: x.isdigit() and 1 <= int(x) <= 5))
            
            if option == 1:
                saved_model_test()
            elif option == 2:
                continue
            elif option == 3:
                continue
            elif option == 4:
                continue
            elif option == 5:
                print('\nSaliendo del programa...')
                break
    except KeyboardInterrupt:
        print('\nSaliendo del programa...')
        exit()

if __name__ == '__main__':
    read_optimal_solution('.\\benchmarks\\CVS\\3-4\\Data3-4.xlsx')