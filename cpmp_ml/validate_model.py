from CPMP.cpmp_ml import generate_random_layout
from CPMP.cpmp_ml import read_file
from CPMP.cpmp_ml import greedy_model
from CPMP.cpmp_ml import greedys
from copy import deepcopy
from statistics import mean
from statistics import median

def validate_model(model, S, H, N, greedy: str = 'greedy_v2', verbose: bool = True, cvs_class=None):
    n = 1000

    lays = []
    if cvs_class is None:
        for i in range(n):
            lays.append(generate_random_layout(S,H,N))
    else:
        n=40
        for i in range(1,n+1):
            lay = read_file(f"benchmarks/CVS/{cvs_class}/data{cvs_class}-{i}.dat",5)
            lays.append(lay)

    lays1 = deepcopy(lays)
    costs1 = greedy_model(model, lays1, S= S, H= H, max_steps=N*2)
    costs2 = greedys(lays, max_steps= 80)

    valid_costs1 = [v for v in costs1 if v!=-1]
    valid_costs2 = [v for v in costs2 if v!=-1]

    results_model = len(valid_costs1) / n * 100.
    results_greedy = len(valid_costs2) / n * 100.

    if len(valid_costs1)>0:
        print(f"success ann model (%): {results_model}") 
        print(f"mean steps: {mean(valid_costs1)}")
        print(f"median steps: {median(valid_costs1)}")
        #print(f"stdesv steps: {stdev(valid_costs1)}")
        print(f"min steps: {min(valid_costs1)}")
        print(f"max steps: {max(valid_costs1)}")
        print('')
    if len(valid_costs2)==0:
        print("success heuristic (%):", results_greedy)
    else:
        print("success heuristic (%):", results_greedy, mean(valid_costs2))
        print(f"mean steps: {mean(valid_costs2)}")
        print(f"median steps: {median(valid_costs2)}")
        #print(f"stdesv steps: {stdev(valid_costs2)}")
        print(f"min steps: {min(valid_costs2)}")
        print(f"max steps: {max(valid_costs2)}")
        print('')

    return results_model, results_greedy