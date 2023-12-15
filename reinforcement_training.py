import numpy as np
import os
from statistics import mean
from copy import deepcopy
from Layers import ConcatenationLayer, Model_CPMP, Reduction, LayerExpandOutput, OutputMultiplication
from keras.models import Model, load_model

os.chdir('CPMP-ML')
import cpmp_ml
from cpmp_ml import generate_random_layout, greedy_model, greedys, read_file, generate_data2
os.chdir('..')

def get_ann_state(layout: cpmp_ml.Layout) -> np.ndarray:
  S=len(layout.stacks) # Cantidad de stacks
  #matriz de stacks
  b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
  for i,j in enumerate(layout.stacks):
     b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
     b[i][0] = layout.is_sorted_stack(i)
  b.shape=(S,(layout.H + 1))
  return b

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
    cpmp_ml.get_ann_state = get_ann_state

    model = load_cpmp_model('models/model_cpmp_5x5_test_v2.h5')

    reinforcement_training(model= model, S= 5, H= 5, MPC= 15, sample_size= 30000, iter= 2, batch_size= 55) 

    model.save('models/model_cpmp_5x5_test_v2.h5')