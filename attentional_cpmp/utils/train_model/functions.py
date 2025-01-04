from keras.src.callbacks import EarlyStopping
from keras.src.models import Model

from typing import Any
import numpy as np

def normal_training(model: Model, 
                    data: dict, 
                    max_samples: int | None = None, 
                    batch_size: int = 32, 
                    epochs: int = 10, 
                    validation_split: float = 0.2,
                    patience: int = 3,
                    monitor: str = 'val_loss',
                    verbose: int = 1) -> dict:
    '''
    Entrana el modelo con los datos de entrenamiento.

    Args:
        model: Modelo de Keras a entrenar.
        data: Diccionario con los datos de entrenamiento, exiten N estados que son diccionarios y posee States y Labels.
        max_samples: Número máximo de muestras a utilizar para el entrenamiento.
        batch_size: Tamaño del lote para el entrenamiento.
        epochs: Número de épocas para el entrenamiento.
        validation_split: Porcentaje de datos de validación.
        patience: Número de épocas sin mejora antes de detener el entrenamiento.
        monitor: Métrica a monitorear para el EarlyStopping.

    Returns:
        history: Historial de métricas del entrenamiento.
    '''
    if max_samples is not None:
        for stack in data:
            data[stack]["States"] = data[stack]["States"][:max_samples]
            data[stack]["Labels"] = data[stack]["Labels"][:max_samples]

    state_history = {}

    for stack in data:
        if verbose == 1: print(f"Entrenando con el estado {stack}...")
        history = model.fit(np.stack(data[stack]["States"]), np.stack(data[stack]["Labels"]), 
                            batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                            callbacks=[EarlyStopping(monitor=monitor, 
                                                     patience=patience, 
                                                     restore_best_weights=True,
                                                     verbose=verbose)],
                            verbose=verbose)
        
        state_history[stack] = {metric_name: history.history[metric_name] for metric_name in history.history.keys()}
        state_history[stack]['epochs'] = [i for i in range(1, len(state_history[stack][monitor])+1)]
    
    return state_history
        

def batch_training(model: Model, 
                   data: dict, 
                   max_samples: int | None = None, 
                   n_subsets: int = 5, 
                   epochs: int = 10,
                   validation_split: float = 0.2,
                   batch_size: int = 32,
                   patience: int = 3,
                   monitor: str = 'val_loss',
                   verbose: int = 1) -> dict:
    '''
    Entrana el modelo con los datos divididos en subconjuntos para cada estado.

    Args:
        model: Modelo de Keras a entrenar.
        data: Diccionario con los datos de entrenamiento, exiten N estados que son diccionarios y posee States y Labels.
        max_samples: Número máximo de muestras a utilizar para el entrenamiento.
        n_subsets: Número de subconjuntos en los que se dividirán los datos.
        metrics: Lista de métricas a registrar durante el entrenamiento.
        epochs: Número de épocas para el entrenamiento.
        validation_split: Porcentaje de datos de validación.
        batch_size: Tamaño del lote para el entrenamiento.
        verbose: Nivel de verbosidad.

    Returns:
        state_history: Diccionario con el historial de métricas por cada estado.
    '''

    if max_samples is not None:
        for stack in data:
            data[stack]["States"] = data[stack]["States"][:max_samples]
            data[stack]["Labels"] = data[stack]["Labels"][:max_samples]

    new_data = split_data(data, n_subsets=n_subsets, states=data.keys())

    if verbose == 1: print("Entrenando con división de datos en subconjuntos...")

    state_history = {}

    max_subsets = max(len(sub_states) for sub_states in new_data.values())

    for subset_idx in range(max_subsets):
        if verbose == 1: print(f"Entrenando con el subconjunto {subset_idx + 1} de cada estado...")
        for state, sub_states in new_data.items():
            if subset_idx < len(sub_states):  # Verificar que el subconjunto existe
                if verbose == 1: print(f"Estado: {state}, Subconjunto: {subset_idx + 1}")
                sub_state = sub_states[subset_idx]
                history = model.fit(
                    sub_state["States"],
                    sub_state["Labels"],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split,
                    callbacks=[EarlyStopping(monitor=monitor, 
                                             patience=patience, 
                                             restore_best_weights=True,
                                             verbose=verbose)],
                )
                for metric, values in history.history.items():
                    if state not in state_history:
                        state_history[state] = {}
                    if metric not in state_history[state]:
                        state_history[state][metric] = values
                    else:
                        state_history[state][metric].extend(values)

    for states in data.keys():
        state_history[states]['epochs'] = [i for i in range(1, len(state_history[states][monitor])+1)]

    return state_history

def split_data(data, n_subsets, states):
    """Divide los datos en N subconjuntos para cada estado, independientemente del tamaño inicial de los datos."""
    new_data = {}

    for state in states:
        total_size = len(data[state]['States'])
        subset_size = total_size // n_subsets  # Tamaño base de cada subconjunto
        remainder = total_size % n_subsets  # Datos sobrantes

        sub_states = []
        sub_labels = []

        start_idx = 0
        for i in range(n_subsets):
            # Ajustar el tamaño del subset para distribuir el sobrante
            current_subset_size = subset_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_subset_size

            sub_states.append(data[state]['States'][start_idx:end_idx])
            sub_labels.append(data[state]['Labels'][start_idx:end_idx])

            start_idx = end_idx

        # Guardar los subconjuntos en el nuevo diccionario
        new_data[state] = [
            {"States": np.stack(sub_state), "Labels": np.stack(sub_label)}
            for sub_state, sub_label in zip(sub_states, sub_labels)
        ]

    return new_data