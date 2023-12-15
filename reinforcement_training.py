import numpy as np
import os
import random
from statistics import mean
from copy import deepcopy
from Layers import ConcatenationLayer, Model_CPMP, Reduction, LayerExpandOutput, OutputMultiplication
from keras.models import Model, load_model
from CPMP import cpmp_ml
from CPMP.cpmp_ml import generate_data2, greedy_model, generate_random_layout
from CPMP.cpmp_ml import greedys, read_file, get_move, permutate_y

def get_ann_state(layout: cpmp_ml.Layout) -> np.ndarray:
    S=len(layout.stacks) # Cantidad de stacks
    #matriz de stacks
    b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
    for i,j in enumerate(layout.stacks):
        b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
        b[i][0] = layout.is_sorted_stack(i)
    b.shape=(S,(layout.H + 1))
    return b

def greedy_model(model, layouts, max_steps=10):
  from keras import backend as K
  costs = -np.ones(len(layouts))

  for steps in range(max_steps):
    x = []
    for i in range(len(layouts)):
      if layouts[i].unsorted_stacks==0: 
        if costs[i] ==-1: costs[i]=steps
        continue
      x.append(get_ann_state(layouts[i]))
    
    if len(x)==0: break
    actions = model.predict(np.array(x), verbose=False)
    K.clear_session()
    k=0
    for i in range(len(layouts)):
      if costs[i] != -1: continue
      act = np.argmax(actions[k])
      move = get_move(act)
      layouts[i].move(move)
      k+=1
  return costs

def generate_data2(
    model,
    S=5,
    H=5,
    N=10,
    sample_size=1000,
    max_steps=20,
    batch_size=1000,
    perms_by_layout=20,
):
    x = []
    y = []

    while True:
        lays = []
        for i in range(batch_size):
            lays.append(generate_random_layout(S, H, N))
            # print ("Layout generado:", lays[i].stacks)

        lays0 = deepcopy(lays)
        costs = greedy_model(model, lays, max_steps=max_steps)

        # lays that cannot be solved by the model
        # lays0 = [lays0[i] for i in range(batch_size) if costs[i]==-1]
        # lays = [lays[i] for i in range(batch_size) if costs[i]==-1]
        # print("Costo obtenido por modelo:", len(lays))

        # for each lay we generate children clays
        clays = []
        for p in range(len(lays)):
            for i in range(S):
                for j in range(S):
                    if i == j:
                        continue
                    clay = deepcopy(lays0[p])
                    clay.move((i, j))
                    clays.append(clay)
            # print("len clays", len(clays))
        # print (f"clays generados {len(clays)}")
        # clays are solved

        ccosts = greedy_model(model, clays, max_steps=max_steps)
        # print("costs", ccosts)

        # f = lambda parent, k: (parent * (S*(S-1))) + k

        # Para cada padre
        for p in range(len(lays)):
            # print(lays[p].stacks)
            # print (ccosts[p*S*(S-1):(p+1)*S*(S-1)])
            A = []
            mincost = np.inf
            for c in range(p * (S * (S - 1)), (p + 1) * (S * (S - 1))):
                if ccosts[c] != -1 and ccosts[c] < mincost:
                    mincost = ccosts[c]

            if costs[p] != -1 and mincost >= costs[p]:
                continue

            for c in range(p * (S * (S - 1)), (p + 1) * (S * (S - 1))):
                if ccosts[c] != -1 and ccosts[c] == mincost:
                    A.append(1)
                else:
                    A.append(0)

            if (
                sum(A) > 0
            ):  # otherwise no action was succesful, we simply discard the data
                for k in range(perms_by_layout):
                    enum_stacks = list(range(S))
                    perm = random.sample(enum_stacks, S)
                    lays0[p].permutate(perm)
                    A = permutate_y(A, S, perm)

                    x.append(get_ann_state(lays0[p]))
                    y.append(deepcopy(A))
                    if len(x) % 100 == 0: print(f'sample_size {len(x)}')
                    if len(x) == sample_size:
                        return x, y

def validate_model(model, S, H, N, verbose: bool = True, cvs_class=None):
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
    costs1 = greedy_model(model, lays1, max_steps=N*2)
    costs2 = greedys(lays)

    valid_costs1 = [v for v in costs1 if v!=-1]
    valid_costs2 = [v for v in costs2 if v!=-1]

    results_model = len(valid_costs1) / n * 100.
    results_greedy = len(valid_costs2) / n * 100.

    if len(valid_costs1)>0:
        print("success ann model (%):", results_model, mean(valid_costs1))
    if len(valid_costs2)==0:
        print("success heuristic (%):", results_greedy)
    else:
        print("success heuristic (%):", results_greedy, mean(valid_costs2))

    return results_model, results_greedy

def reinforcement_training(model: Model, S: int = 5, H: int = 5, MPC: int = 15, 
                           sample_size: int = 100000, iter: int = 5, max_steps: int = 30, epochs: int = 5, 
                           batch_size: int = 20, verbose: bool = True, perms_by_layout: int = 1) -> None:
    for i in range(iter):
        if verbose: print(f"Step {i + 1}")

        data, labels = generate_data2(model= model, S= S, H= H, N= MPC, sample_size= sample_size, max_steps= max_steps
                                      , batch_size= batch_size, perms_by_layout= perms_by_layout)

        data = np.stack(data)
        labels = np.stack(labels)

        model.fit(data, labels, epochs= epochs, verbose= verbose)
        results_model, results_greedy = validate_model(model, S, H, MPC, verbose= verbose)

        del data, labels

        if verbose: print('')
        if results_model > 90.0: break

def load_cpmp_model(name: str) -> Model:
    custom_objects = {'Model_CPMP': Model_CPMP, 
                      'OutputMultiplication': OutputMultiplication,
                      'LayerExpandOutput': LayerExpandOutput,
                      'ConcatenationLayer': ConcatenationLayer,
                      'Reduction': Reduction}
    
    model = load_model(name, custom_objects= custom_objects)

    return model

if __name__ == '__main__':
    cpmp_ml.generate_data2 = generate_data2
    cpmp_ml.greedy_model = greedy_model
    cpmp_ml.get_ann_state = get_ann_state

    model = load_cpmp_model('models/model_cpmp_5x5_test.h5')

    reinforcement_training(model= model, S= 5, H= 5, MPC= 15, sample_size= 30000, iter= 2, batch_size= 55) 

    model.save('models/model_cpmp_5x5.h5')