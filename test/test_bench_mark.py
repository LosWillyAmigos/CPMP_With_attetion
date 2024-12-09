from attentional_cpmp.utils import load_data_mongo
from attentional_cpmp.utils import connect_to_server
from attentional_cpmp.model import load_cpmp_model
from attentional_cpmp.optimizer import BeamSearch
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.optimizer import OptimizerStrategy, GreedyModel, GreedyV1, GreedyV2
from cpmp_ml.utils import Layout
from keras.models import Model
from statistics import mean
import pandas as pd
import numpy as np
import json
import os

def read_benchmark_file(route: str) -> Layout:
    H = 0

    with open(route) as file:
        S, N = (int(x) for x in next(file).split())

        stacks = []
        for line in file:
            stack = [int(x) for x in line.split()[1::]]
            stacks.append(stack)
            H = max(H, len(stack))

        lay = Layout(stacks, H)

    return lay, S, N

def read_files(dir_path: str) -> list:
    lays = []
    opt = 0

    for root, dirs, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('.bay'): 
                lay, S, N = read_benchmark_file(root + filename)
                lays.append(lay)

            elif filename.endswith('.txt'): opt = int(open(root + filename).readline())

    return np.stack(lays), S, N, opt

def create_flags(lists_costs: list[list]) -> list:
    filter_costs = [True for _ in range(len(lists_costs[0]))]

    for costs in lists_costs:
        for i in range(len(costs)):
            if costs[i] == -1:
                filter_costs[i] = False

    return filter_costs

def filter_costs(lists_costs: list[list]) -> np.ndarray:
    flags = create_flags(lists_costs)
    filter_opt = []

    for i in range(len(lists_costs)):
        filter_opt.append([])
        for j in range(len(lists_costs[i])):
            if flags[j]: filter_opt[i].append(lists_costs[i][j])

    return filter_opt

def solve_problems(optimizers: list[OptimizerStrategy], lays: np.ndarray[Layout], N: int) -> list[list]:
    costs_opt = []
    
    for opt in optimizers:
        cost = opt.solve(lays, max_steps= N * 2)[0]
        costs_opt.append(cost)

    return costs_opt

def filter_comparative(costs_opt: list[list], del_opt: list) -> list:
    return [cost for i, cost in enumerate(costs_opt) if i not in del_opt]

def build_benchmark_df(df: pd.DataFrame, instance_name: str, 
                       S: int, H: int, N: int, opt: int,
                       model_params: dict, total_params, level_nodes: int, threshold: float,
                       costs_opt: list[list], costs_comp: list[list]) -> pd.DataFrame:
    init_info = [instance_name, S, H, N, opt, 
                 model_params['key_dim'], model_params['value_dim'], model_params['num_heads'], 
                 model_params['num_stacks'], model_params['epsilon'], total_params,
                 level_nodes, threshold]
    steps = []
    
    for i in range(len(costs_opt)):
        steps += [mean(costs_opt[i])]

    for i in range(len(costs_comp)):
        steps += [mean(costs_comp[i])]

    df.loc[len(df)] = init_info + steps

    return df

df_benchmarks = pd.read_excel('./test/experiments/benchmarks.xlsx')

for i in range(1,33):
    lays, S, N, opt = read_files(f'./CPMP/benchmarks/BF/BF{i}/')
    model = load_cpmp_model(f'./models/Sx{S}/model_Sx{S}.h5')
    model_params = json.load(open(f'./models/Sx{S}/model_Sx{S}.json'))
    H = lays[0].shape[1] - 1
    
    opt_model = GreedyModel(model, AttentionModel())
    opt_BS = BeamSearch(model, AttentionModel(), H, S, 0.5)
    opt_greedy_v1 = GreedyV1()
    opt_greedy_v2 = GreedyV2()

    costs_opt = solve_problems([opt_model, opt_BS, opt_greedy_v1, opt_greedy_v2], lays, N)
    costs_comparative = filter_comparative(costs_opt, [2])
    costs_comparative = filter_costs(costs_comparative)

    df_benchmarks = build_benchmark_df(df_benchmarks, f'BF{i}', S, H, N, opt, model_params, model.count_params(), S, 0.5, costs_opt, costs_comparative)
