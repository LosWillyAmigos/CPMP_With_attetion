from keras.models import Model
from copy import deepcopy
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils import generate_random_layout

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
from numpy import mean, median

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
            print(f"Optimizer {i+1} could not solved any of the {len(final_costs[i])}/{sample_size} problems considered")
        else:
            if calculate_only_solved:
                show_only_solved(final_costs[i], len(final_costs[i]), sample_size)
            else: show_all_solved(final_costs[i], number_of_decimals, sample_size)

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
                
            
def show_all_solved(costs:list, number_of_decimals:int, sample_size:int):
    valid_costs = [v for v in costs if v!=-1]
    results = len(valid_costs) / sample_size * 100.
    print(f"Problems solved (%): {round(results, number_of_decimals)}%")
    print(f"Mean steps: {mean(valid_costs)}")
    print(f"Median steps: {median(valid_costs)}")
    print(f"Min steps: {min(valid_costs)}")
    print(f"Max steps: {max(valid_costs)}")
    print('')

def show_only_solved(costs:list, number_of_solved:int, sample_size:int):
    valid_costs = [v for v in costs if v!=-1]
    print(f'Number of problems solved: {number_of_solved}/{sample_size}')
    print(f"Mean steps: {mean(valid_costs)}")
    print(f"Median steps: {median(valid_costs)}")
    print(f"Min steps: {min(valid_costs)}")
    print(f"Max steps: {max(valid_costs)}")
    print('')


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

    lays_N = [[generate_random_layout(S, H, n) for _ in range(sample_size)] for n in x]

    for n in x:
        costs = optimizer.solve(lays=np.array(lays_N[n - (S * 2)]), **kwargs)
        valid_costs = [v for v in costs if v != -1]
        results_model = len(valid_costs) / sample_size * 100.0
        y.append(results_model)
        mean_list.append(np.mean(valid_costs))
        median_list.append(np.median(valid_costs))
        min_list.append(np.min(valid_costs))
        max_list.append(np.max(valid_costs))

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

    # Guardar resultados en un Excel
    if excel_path is not None:
        results_df = pd.DataFrame({"N - Cantidad de contenedores": x, "Porcentaje de acierto": y})
        excel_name = f"{model_name}_S:{S}_H:{H}.xlsx"
        results_df.to_excel(os.path.join(os.path.dirname(excel_path), excel_name), index=False)

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

    if N is not None:
        for s in S:
            lays_S.append([generate_random_layout(s, H, N) for _ in range(sample_size)])
    elif N is None:
        for s in S:
            lays_S.append([generate_random_layout(s, H, random.randint(S * 2, (S * (H - 2)))) for _ in range(sample_size)])

    for s in range(len(S)):
        kwargs["max_steps"] = len(S[s]) * (H - 2) * 2
        costs, _ = optimizer.solve(lays=np.array(lays_S[s]), **kwargs)
        valid_costs = [v for v in costs if v != -1]
        results_model = len(valid_costs) / sample_size * 100.0
        y.append(results_model)
        mean_list.append(np.mean(valid_costs))
        median_list.append(np.median(valid_costs))
        min_list.append(np.min(valid_costs))
        max_list.append(np.max(valid_costs))

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

    # Guardar resultados en un Excel
    if excel_path is not None:
        results_df = pd.DataFrame({"S - pilas": S, "Porcentaje de acierto": y})
        excel_name = f"{model_name}_some_Sx{H}.xlsx"
        results_df.to_excel(os.path.join(os.path.dirname(excel_path), excel_name), index=False)