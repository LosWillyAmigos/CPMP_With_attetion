from keras.models import Model
from copy import deepcopy
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils import generate_random_layout

import numpy as np
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
    