import os
import numpy as np
from copy import deepcopy
from CPMP.cpmp_ml import generate_random_layout
from CPMP.cpmp_ml import Layout
from cpmp_ml.model import load_cpmp_model

ERR_MSG_CYCLE = 'Se ha detectado un ciclo!'
ERR_MSG_MOVE = 'Se ha intentado realizar un movimiento que no es posible'
SUCCESS_MSG = 'Problema resuelto!'

def get_move(act, S=5):
  k=0
  for i in range(S):
    for j in range(S):
      if(i==j): continue
      if k==act: return (i,j)
      k+=1

def get_ann_state_v2(layout: Layout) -> np.ndarray:
    """
    The purpose of this function is to prepare the
    data of a CPMP problem state so that it can be
    read by a neural network.

    Input: 
        layout (cpmp.Layout): Current state of the CPMP 
                              problem.
    
    Return:
        ndarray: matrix with normalized data.
    """
    S=len(layout.stacks) # Cantidad de stacks
    #matriz de stacks
    b = 2. * np.ones([S,layout.H + 1]) # Matriz normalizada
    for i,j in enumerate(layout.stacks):
        b[i][layout.H-len(j) + 1:] = [k/layout.total_elements for k in j]
        b[i][0] = layout.is_sorted_stack(i)
    b.shape=(S,(layout.H + 1))
    return b

def clear_terminal():
    os.system('cls')

def detect_cycles(states, state):
    size = len(states)
    #stack_size = len(state.stacks)

    for i in range(size):
        if states[i].stacks == state.stacks: return True
    
    return False

def resolve_cpmp_problem(model, stack, height, conteiners):
    states = []
    state = generate_random_layout(S= stack, H= height, N= conteiners)

    while(True):
        states.append(deepcopy(state))
        if state.unsorted_stacks == 0: break
        nn_state = get_ann_state_v2(state)

        act = model.predict(np.stack([nn_state]), verbose= False)
        k = np.argmax(act)
        move = get_move(k, S= stack, H= stack)

        print(state.stacks)
        container = state.move(move)
        print(state.stacks, '\n', container)

        if container is None:
            states.append(deepcopy(state))
            return states, 1, None
        elif detect_cycles(states, state):
            states.append(deepcopy(state))
            return states, 2, move

    return states, 0, None

def show_stack(stack, height):
    size = len(stack)

    if size != 0: print(f'[{stack[0]:2d}', end= ' ')
    else: print(f'[{0:2d}', end= ' ')

    for i in range(1, height - 1):
        if i >= size:
            print(f'{0:2d}', end= ' ')
        else:
            print(f'{stack[i]:2d}', end= ' ')

    if size == height: print(f'{stack[height - 1]:2d}', end= ']')
    else: print(f'{0:2d}', end=']')


def show_matrixs(states, stacks, height):
    size = len(states)

    for j in range(stacks):
        for k in range(size):
            show_stack(states[k].stacks[j], height)
            print(' ', end= ' ')
        
        print('')

def visualize(model, stack, height, conteiners):
    states, err, move = resolve_cpmp_problem(model, stack, height, conteiners)
    size = len(states)

    show_matrixs(states, stack, height)

    if err == 2: print(ERR_MSG_CYCLE)
    elif err == 1: print(f'{ERR_MSG_MOVE}  move: {move}')
    else: print(SUCCESS_MSG)

def repeat():
    while(True):
        print('Desea realizar una nueva prueba? (S/N)')
        act = input()

        if act.isalpha():
            if act.lower() == 's': return True
            if act.lower() == 'n': return False
        
        clear_terminal()
        print('Opción invalida, vuelva a intentar.')

def is_BG_action(self, action):
    s_o = action[0]; s_d = action[1]
    if (self.is_sorted_stack(s_o)==False
    and self.is_sorted_stack(s_d)==True
    and self.gvalue(s_o) <= self.gvalue(s_d)):
      return True

    else: return False

def visualization():
    file_name = input('Indique el nombre del modelo que desea cargar: ')
    model  = load_cpmp_model(file_name)
    
    if model is None: 
        print('Modelo no encontrado\n')
    else:
        print('Modelo cargado con éxito\n')

    height_size = model.layers[0].input_shape[0][2] - 1
    stack_size = int(input('Indique la cantidad de stacks: '))
    print(f'Indique la cantidad de contenedores para problemas {stack_size}x{height_size}:', end= ' ')
    conteiner_size = int(input())

    while(True):
        clear_terminal()
        visualize(model, stack_size, height_size, conteiner_size)

        if not repeat(): break