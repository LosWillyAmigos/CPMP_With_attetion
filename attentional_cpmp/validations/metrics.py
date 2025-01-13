from keras.models import Model
from copy import deepcopy
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils import generate_random_layout
from numpy import mean, median
from keras.src.metrics import Metric, MeanSquaredError

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import time

    
class PercentageSolved(Metric):
    def __init__(self, name="percentage_solved", threshold = 0.98, **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.thresold = threshold
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_binary = tf.cast(y_pred > self.thresold, tf.float32)
        corrects = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32), axis=-1)
        percentage = corrects / tf.cast(tf.shape(y_true)[-1], tf.float32)
        
        self.total.assign_add(tf.reduce_sum(percentage))
        self.count.assign_add(tf.cast(tf.size(percentage), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)


def percentage_solved(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    correctas = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_binary), tf.float32), axis=-1)
    porcentaje = correctas / tf.cast(tf.shape(y_true)[-1], tf.float32)
    return tf.reduce_mean(porcentaje)

def validation_per_stack(optimizer: OptimizerStrategy,
                         S: list[int],
                         H: int,
                         N: list[int] | None = None,
                         sample_size: int = 1000,
                         median_percentage: bool = False,):
    '''
    Función de validación de estados aleatorios para un conjunto de pilas.

    Args:
        optimizer (OptimizerStrategy): Optimizador a validar.
        S (list[int]): Lista de estados a validar.
        H (int): Altura de las pilas.
        N (list[int], optional): Lista de cantidad de contenedores. Defaults to None.
        sample_size (int, optional): Tamaño de la muestra. Defaults to 1000.
        median_percentage (bool, optional): Calcular la mediana de los porcentajes. Defaults to False.

    Returns:
        float | list[float]: Promedio de Porcentajes de aciertos. Si median=True, retorna la mediana de los porcentajes, en caso
        contrario retorna la lista de lo porcentajes de los estados.
    '''
    
    if N is None: N = [s*(H-2) for s in S]

    percentages = []
    
    for s, n in zip(S, N):
        lays = [generate_random_layout(s, H, n) for _ in range(sample_size)]
        costs, _ = optimizer.solve(lays=np.array(lays), max_steps=s*(H-2)*2)
        valid_costs = [v for v in costs if v!=-1]
        results = len(valid_costs) / sample_size * 100.
        percentages.append(results)

    if median_percentage:
        return np.median(percentages)
    return percentages



def validate_optimizers(optimizers: list[OptimizerStrategy],
                   S: int, 
                   H: int, 
                   N: int, 
                   sample_size: int,
                   benchmark_csv: str = None, 
                   calculate_only_solved: bool = False,
                   number_of_decimals: int = 2,
                   **kwargs) -> dict:

    if benchmark_csv is None:
        lays = [generate_random_layout(S, H, N) for _ in range(sample_size)]

    costs = []

    for optimizer in optimizers:
        copy_states = deepcopy(lays)
        cost, _ = optimizer.solve(lays=np.array(copy_states), **kwargs)
        costs.append(cost)

    if calculate_only_solved:
        counter_solved = calculate_solved(costs, sample_size)
        final_costs = get_final_costs(costs, counter_solved, sample_size, len(optimizers))
    else:
        final_costs = costs
        
    for i in range(len(optimizers)):
        print(f'************ Optimizer {i+1} ****************')
        if len(final_costs[i]) == 0:
            print(f'Optimizer {i+1} could not solved any of the {len(final_costs[i])}/{sample_size} problems considered')
        else:
            if calculate_only_solved:
                show_only_solved(final_costs[i], len(final_costs[i]), sample_size)
            else: show_all_solved(final_costs[i], number_of_decimals, sample_size)

def validation_optimizer_per_container(optimizers: list[OptimizerStrategy],
                                       optimizers_name: list[str],
                                       S: list,
                                       H: int,
                                       sample_size: int = 1000,
                                       output_dir: str = "output/",
                                       calculate_only_solved: bool = False,
                                       hyperparameter_name: str = "model"):
    for state in S:
        # Guardar la gráfica para cada estado
        plot_path = f'{output_dir}plots/N_CONTAINERS/'
        excel_path = f'{output_dir}excels/N_CONTAINERS/'
        print(f'Generando gráfica para estado {state}...')
        validation_optimizer_per_C(optimizers=optimizers,
                                 optimizers_name=optimizers_name,
                                 S=int(state),
                                 H=H,
                                 sample_size=sample_size,
                                 excel_path=excel_path,
                                 calculate_only_solved=calculate_only_solved,
                                 path_plot=plot_path,
                                 create_plot=True,
                                 model_name=hyperparameter_name,
                                 max_steps=int(state) * (H - 2) * 2)


def validation_optimizer_per_C(optimizers: list[OptimizerStrategy],
                            optimizers_name: list[str],
                            S: int,
                            H: int,
                            sample_size: int = 1000,
                            excel_path: str | None = None,
                            calculate_only_solved: bool = False,
                            create_plot: bool = False,
                            model_name: str = "model",
                            path_plot: str = "plot.png",
                            **kwargs):
    
    os.makedirs(path_plot, exist_ok=True)
    os.makedirs(excel_path, exist_ok=True)
    
    x = [i for i in range(S * 2, (S * (H - 2)) + 1)]  # Límites de S*2 hasta S*(H-2)

    lays_N = [[generate_random_layout(S, H, n) for _ in range(sample_size)] for n in x]

    excel_content = {}
    for name in optimizers_name:
        excel_content[name] = {
            "State" : x,
            "Percentage" : [],
            "Mean" : [],
            "Median" : [],
            "Min" : [],
            "Max" : [],
            "Time" : [],
            "Solved" : [],
            "Size Sample" : []
        }

    if calculate_only_solved: 
        all_optimizers = {
            "State" : x,
            "Percentage" : [],
            "Solved" : [],
            "Size Sample" : []
        }
        for optimizer in optimizers_name:
            all_optimizers[f'{optimizer} - Mean'] = []
            all_optimizers[f'{optimizer} - Median'] = []
            all_optimizers[f'{optimizer} - Min'] = []
            all_optimizers[f'{optimizer} - Max'] = []
        
    for n in x:
        costs = []
        for i, optimizer in enumerate(optimizers):
            copy_states = deepcopy(lays_N[n - (S * 2)])
            start_time = time.time()
            cost, _ = optimizer.solve(lays=np.array(copy_states), **kwargs)
            end_time = time.time()
            delta = end_time - start_time
            costs.append(cost)
            excel_content[optimizers_name[i]]["Time"].append(f'{delta:.12f}')

        for i, optimizer in enumerate(optimizers):
            results, mean_steps, median_steps, min_steps, max_steps = get_statics(costs[i], 6, sample_size)
            excel_content[optimizers_name[i]]["Percentage"].append(results)
            excel_content[optimizers_name[i]]["Mean"].append(mean_steps)
            excel_content[optimizers_name[i]]["Median"].append(median_steps)
            excel_content[optimizers_name[i]]["Min"].append(min_steps)
            excel_content[optimizers_name[i]]["Max"].append(max_steps)
            excel_content[optimizers_name[i]]["Solved"].append(len([v for v in costs[i] if v!=-1]))
            excel_content[optimizers_name[i]]["Size Sample"].append(sample_size)

        if calculate_only_solved: 
            counter_solved = calculate_solved(costs, sample_size)
            final_costs = get_final_costs(costs, counter_solved, sample_size, len(optimizers))

            
            all_optimizers["Percentage"].append(len([v for v in final_costs[0] if v!=-1]) / sample_size * 100)
            all_optimizers["Solved"].append(len(final_costs[0]))
            all_optimizers["Size Sample"].append(sample_size)
            for i, optimizer in enumerate(optimizers_name):
                results, mean_steps, median_steps, min_steps, max_steps = get_statics(final_costs[i], 6, sample_size)
                all_optimizers[f'{optimizer} - Mean'].append(mean_steps)
                all_optimizers[f'{optimizer} - Median'].append(median_steps)
                all_optimizers[f'{optimizer} - Min'].append(min_steps)
                all_optimizers[f'{optimizer} - Max'].append(max_steps)

    # Crear un único archivo Excel con múltiples hojas
    with pd.ExcelWriter(f'{excel_path}{model_name}_STATE_{S}_N_CONTAINERS.xlsx', engine='openpyxl') as writer:
        for name in optimizers_name:
            df = pd.DataFrame(excel_content[name])
            df.to_excel(writer, sheet_name=name, index=False)
        if calculate_only_solved:
            df = pd.DataFrame(all_optimizers)
            df.to_excel(writer, sheet_name="All optimizers", index=False)


    if create_plot:
        for i, optimizer in enumerate(optimizers):
            plt.plot(x, excel_content[optimizers_name[i]]["Percentage"], marker='o')
            plt.xlabel('N - Cantidad de contenedores')
            plt.ylabel('Porcentaje')
            plt.title(f'Porcentaje de acierto en relación N - {model_name} - {optimizers_name[i]}')
            plt.grid(True)
            plt.savefig(f'{path_plot}{model_name}_{optimizers_name[i]}_TO_STATE_{S}_N_CONTAINERS_PLOT.png')
            plt.close()

        if calculate_only_solved:
            plt.plot(x, all_optimizers["Percentage"], marker='o')
            plt.xlabel('N - Cantidad de contenedores')
            plt.ylabel('Porcentaje')
            plt.title(f'Porcentaje de acierto en relación N - {model_name} - Para todos los optimizadores')
            plt.grid(True)
            plt.savefig(f'{path_plot}{model_name}_ALL_OPTIMIZERS_TO_STATE_{S}_N_CONTAINERS_PLOT.png')
            plt.close()

def validation_optimizer_per_stack(optimizers: list[OptimizerStrategy],
                                       optimizers_name: list[str],
                                       S: list,
                                       H: int,
                                       N: int | None = None,
                                       sample_size: int = 1000,
                                       output_dir: str = "output/",
                                       calculate_only_solved: bool = False,
                                       hyperparameter_name: str = "model"):
    plot_path = f'{output_dir}plots/S_STACKS/'
    excel_path = f'{output_dir}excels/S_STACKS/'
    print(f'Generando gráfica para estado {S}...')
    validation_optimizer_per_S(optimizers=optimizers,
                                optimizers_name=optimizers_name,
                                S=[int(state) for state in S],
                                H=H,
                                N=N,
                                sample_size=sample_size,
                                excel_path=excel_path,
                                calculate_only_solved=calculate_only_solved,
                                path_plot=plot_path,
                                create_plot=True,
                                model_name=hyperparameter_name)
            
def validation_optimizer_per_S(optimizers: list[OptimizerStrategy],
                            optimizers_name: list[str],
                            S: list[int],
                            H: int,
                            N: int | None = None,
                            sample_size: int = 1000,
                            excel_path: str | None = None,
                            calculate_only_solved: bool = False,
                            create_plot: bool = False,
                            model_name: str = "model",
                            path_plot: str = "plot.png",
                            **kwargs):
    
    os.makedirs(path_plot, exist_ok=True)
    os.makedirs(excel_path, exist_ok=True)
    lays_S = []

    S.sort()

    if N is not None:
        for s in S:
            lays_S.append([generate_random_layout(s, H, s*(H-2)) for _ in range(sample_size)])
    elif N is None:
        for s in S:
            lays_S.append([generate_random_layout(s, H, random.randint(s * 2, (s * (H - 2)))) for _ in range(sample_size)])

    excel_content = {}
    for name in optimizers_name:
        excel_content[name] = {
            "State - S" : S,
            "Percentage" : [],
            "Mean" : [],
            "Median" : [],
            "Min" : [],
            "Max" : [],
            "Time" : [],
            "Solved" : [],
            "Size Sample" : []
        }

    if calculate_only_solved: 
        all_optimizers = {
            "State" : S,
            "Percentage" : [],
            "Solved" : [],
            "Size Sample" : []
        }
        for optimizer in optimizers_name:
            all_optimizers[f'{optimizer} - Mean'] = []
            all_optimizers[f'{optimizer} - Median'] = []
            all_optimizers[f'{optimizer} - Min'] = []
            all_optimizers[f'{optimizer} - Max'] = []

    for s in range(len(S)):
        costs = []
        for i, optimizer in enumerate(optimizers):
            kwargs["max_steps"] = S[s] * (H - 2) * 2
            copy_states = deepcopy(lays_S[s])
            start_time = time.time()
            cost, _ = optimizer.solve(lays=np.array(copy_states), **kwargs)
            end_time = time.time()
            delta = end_time - start_time
            costs.append(cost)
            excel_content[optimizers_name[i]]["Time"].append(f'{delta:.12f}')

        for i, optimizer in enumerate(optimizers):
            results, mean_steps, median_steps, min_steps, max_steps = get_statics(costs[i], 6, sample_size)
            excel_content[optimizers_name[i]]["Percentage"].append(results)
            excel_content[optimizers_name[i]]["Mean"].append(mean_steps)
            excel_content[optimizers_name[i]]["Median"].append(median_steps)
            excel_content[optimizers_name[i]]["Min"].append(min_steps)
            excel_content[optimizers_name[i]]["Max"].append(max_steps)
            excel_content[optimizers_name[i]]["Solved"].append(len([v for v in costs[i] if v!=-1]))
            excel_content[optimizers_name[i]]["Size Sample"].append(sample_size)

        if calculate_only_solved: 
            counter_solved = calculate_solved(costs, sample_size)
            final_costs = get_final_costs(costs, counter_solved, sample_size, len(optimizers))
            
            all_optimizers["Percentage"].append(len([v for v in final_costs[0] if v!=-1]) / sample_size * 100)
            all_optimizers["Solved"].append(len(final_costs[0]))
            all_optimizers["Size Sample"].append(sample_size)
            for i, optimizer in enumerate(optimizers_name):
                results, mean_steps, median_steps, min_steps, max_steps = get_statics(final_costs[i], 6, sample_size)
                all_optimizers[f'{optimizer} - Mean'].append(mean_steps)
                all_optimizers[f'{optimizer} - Median'].append(median_steps)
                all_optimizers[f'{optimizer} - Min'].append(min_steps)
                all_optimizers[f'{optimizer} - Max'].append(max_steps)

    # Crear un único archivo Excel con múltiples hojas
    with pd.ExcelWriter(f'{excel_path}{model_name}_STATE_{S}_S_STACKS.xlsx', engine='openpyxl') as writer:
        for name in optimizers_name:
            df = pd.DataFrame(excel_content[name])
            df.to_excel(writer, sheet_name=name, index=False)
        if calculate_only_solved:
            df = pd.DataFrame(all_optimizers)
            df.to_excel(writer, sheet_name="All optimizers", index=False)


    if create_plot:
        for i, optimizer in enumerate(optimizers):
            plt.plot(S, excel_content[optimizers_name[i]]["Percentage"], marker='o')
            plt.xlabel('S -STACKS')
            plt.ylabel('Porcentaje')
            plt.title(f'Porcentaje de acierto en relación S - {model_name} - {optimizers_name[i]}')
            plt.grid(True)
            plt.savefig(f'{path_plot}{model_name}_{optimizers_name[i]}_TO_STATE_{S}_S_STACKS_PLOT.png')
            plt.close()
        if calculate_only_solved:
            plt.plot(S, all_optimizers["Percentage"], marker='o')
            plt.xlabel('S -STACKS')
            plt.ylabel('Porcentaje')
            plt.title(f'Porcentaje de acierto en relación S - {model_name} - Para todos los optimizadores')
            plt.grid(True)
            plt.savefig(f'{path_plot}{model_name}_ALL_OPTIMIZERS_TO_STATE_{S}_S_STACKS_PLOT.png')
            plt.close()


def percentage_per_container(optimizer: OptimizerStrategy, 
                             S: int, 
                             H: int, 
                             sample_size: int = 1000,
                             save_path: str ="plot.png", 
                             excel_path: str | None = None, 
                             model_name: str ="model",
                             **kwargs):
    """Genera un gráfico del porcentaje de aciertos por cantidad de contenedores y lo guarda. También guarda los resultados en un Excel."""
    x = [i for i in range(S * 2, (S * (H - 2)) + 1)]  # Límites de S*2 hasta S*(H-2)
    y = []  # Resultados

    mean_list = []
    median_list = []
    min_list = []
    max_list = []
    times_list = []

    lays_N = [[generate_random_layout(S, H, n) for _ in range(sample_size)] for n in x]

    for n in x:
        start_time = time.time()
        costs, _= optimizer.solve(lays=np.array(lays_N[n - (S * 2)]), **kwargs)
        end_time = time.time()
        delta = end_time - start_time
        times_list.append(f'{delta:.12f}')
        results, mean_steps, median_steps, min_steps, max_steps = get_statics(costs, 6, sample_size)
        y.append(results)
        mean_list.append(mean_steps)
        median_list.append(median_steps)
        min_list.append(min_steps)
        max_list.append(max_steps)


    # Crear el gráfico de línea
    plt.plot(x, y, marker='o')

    # Agregar etiquetas y título
    plt.xlabel('N - Cantidad de contenedores')
    plt.ylabel('Porcentaje')
    plt.title(f'Porcentaje de acierto en relación N - {model_name}')

    # Mostrar y guardar el gráfico
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    create_dataframe({f'N - Cantidad de contenedores - {S}x{H}': x, 
                    "Porcentaje de acierto": y,
                    "Media": mean_list,
                    "Mediana": median_list,
                    "Mínimo": min_list,
                    "Máximo": max_list},
                    excel_path, f'{model_name}_STATE_{S}_N_CONTAINERS.xlsx', H)

def percentage_per_S(optimizer: OptimizerStrategy, 
                    S: list[int], 
                    H: int, 
                    N: int | None = None,
                    sample_size: int = 1000,
                    save_path: str ="plot.png", 
                    excel_path: str | None = None, 
                    model_name: str ="model",
                    **kwargs):

    lays_S = []
    y = []

    mean_list = []
    median_list = []
    min_list = []
    max_list = []
    times_list = []

    if N is not None:
        for s in S:
            lays_S.append([generate_random_layout(s, H, N) for _ in range(sample_size)])
    elif N is None:
        for s in S:
            lays_S.append([generate_random_layout(s, H, random.randint(s * 2, (s * (H - 2)))) for _ in range(sample_size)])

    for s in range(len(S)):
        kwargs["max_steps"] = S[s] * (H - 2) * 2
        start_time = time.time()
        costs, _ = optimizer.solve(lays=np.array(lays_S[s]), **kwargs)
        end_time = time.time()
        delta = end_time - start_time
        times_list.append(f'{delta:.12f}')
        results, mean_steps, median_steps, min_steps, max_steps = get_statics(costs, 6, sample_size)
        y.append(results)
        mean_list.append(mean_steps)
        median_list.append(median_steps)
        min_list.append(min_steps)
        max_list.append(max_steps)


    # Crear el gráfico de línea
    plt.plot(S, y, marker='o')

    # Agregar etiquetas y título
    plt.xlabel('N - Cantidad de contenedores')
    plt.ylabel('Porcentaje')
    plt.title(f'Porcentaje de acierto en relación S - {model_name}')

    # Mostrar y guardar el gráfico
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    create_dataframe({"S - pilas": S, 
                    "Porcentaje de acierto": y,
                    "Media": mean_list,
                    "Mediana": median_list,
                    "Mínimo": min_list,
                    "Máximo": max_list},
                    excel_path, f'{model_name}_some_Sx{H}.xlsx', H)


def show_all_solved(costs:list, number_of_decimals:int, sample_size:int):
    valid_costs = [v for v in costs if v!=-1]
    results = len(valid_costs) / sample_size * 100.
    print(f'Problems solved (%): {round(results, number_of_decimals)}%')
    print(f'Mean steps: {mean(valid_costs)}')
    print(f'Median steps: {median(valid_costs)}')
    print(f'Min steps: {min(valid_costs)}')
    print(f'Max steps: {max(valid_costs)}')
    print('')

def show_only_solved(costs:list, number_of_solved:int, sample_size:int):
    valid_costs = [v for v in costs if v!=-1]
    print(f'Number of problems solved: {number_of_solved}/{sample_size}')
    print(f'Mean steps: {mean(valid_costs)}')
    print(f'Median steps: {median(valid_costs)}')
    print(f'Min steps: {min(valid_costs)}')
    print(f'Max steps: {max(valid_costs)}')
    print('')


def get_statics(costs: list, number_of_decimals: int, sample_size: int):
    valid_costs = [v for v in costs if v!=-1]
    if len(valid_costs) == 0: return 0, 0, 0, 0, 0

    results = len(valid_costs) / sample_size * 100.
    mean_steps = np.mean(valid_costs)
    median_steps = np.median(valid_costs)
    min_steps = np.min(valid_costs)
    max_steps = np.max(valid_costs)
    return results, mean_steps, median_steps, min_steps, max_steps


def calculate_solved(costs: list, sample_size: int):
    counter = []
    for index_state in range(sample_size):
            counter.append(search_unsolved(costs, index_state))
    return counter

def search_unsolved(costs: list, index):
    for cost in costs:
        if cost[index] == -1: 
            return False
    return True

def get_final_costs(costs: list[list], solveds: list, sample_size: int, num_optimizer):
    final_costs = [ [] for _ in range(num_optimizer)]
    for index_solved in range(sample_size):
        if solveds[index_solved]:
            for index_cost in range(num_optimizer):
                final_costs[index_cost].append(costs[index_cost][index_solved])

    return final_costs

def get_final_cost(costs: list, solveds: list, sample_size: int):
    final_costs = []
    for index_solved in range(sample_size):
        if solveds[index_solved]:
            final_costs.append(costs[index_solved])

    return final_costs

def create_dataframe(dictionary, excel_path, model_name, H):
    # Guardar resultados en un Excel
    if excel_path is not None:
        results_df = pd.DataFrame(dictionary)
        excel_name = model_name
        results_df.to_excel(os.path.join(os.path.dirname(excel_path), excel_name), index=False)