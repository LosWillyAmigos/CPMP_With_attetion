from attentional_cpmp.model import create_model
from attentional_cpmp.utils.data_saving.data_json import load_data_from_json
from cpmp_ml.validations import validate_model
from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.utils.adapters import AttentionModel
import numpy as np
import json
import argparse

def get_params(dictionary, H):

    new_dictionary = {}

    new_dictionary['H'] = H
    new_dictionary['metrics'] = ['mae', 'mse', 'accuracy']
    new_dictionary['optimizer'] = 'Adam'
    new_dictionary['num_stacks'] = dictionary['num_stacks']
    new_dictionary['num_heads'] = dictionary['num_heads']
    new_dictionary['epsilon'] = dictionary['epsilon']
    new_dictionary['key_dim'] = dictionary['key_dim']
    new_dictionary['value_dim'] = dictionary['value_dim']
    new_dictionary['dropout'] = dictionary['dropout']
    new_dictionary['rate'] = dictionary['rate']
    new_dictionary['activation_hide'] = dictionary['activation_hide']
    new_dictionary['activation_feed'] = dictionary['activation_feed']
    new_dictionary['n_dropout_hide'] = dictionary['n_dropout_hide']
    new_dictionary['n_dropout_feed'] = dictionary['n_dropout_feed']
    new_dictionary['list_neurons_feed'] = dictionary['list_neurons_feed']
    new_dictionary['list_neurons_hide'] = dictionary['list_neurons_hide']

    return new_dictionary

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ejemplo de argparse con parámetros obligatorios y opcionales")

    parser.add_argument('dir_params',
                        type=str, 
                        help="Ruta de los parametros")
    parser.add_argument('dir_data',
                         type=str, 
                         help="Ruta de los datos.")
    parser.add_argument('H',
                         type=int, 
                         help="Dimensión de los datos a recuperar.")
    parser.add_argument('S',
                         type=int, 
                         help="Dimensión de los estados a testear.")
    parser.add_argument('percentage',
                         type=float, 
                         help="Porcentaje de los datos para la validación.")
    
    parser.add_argument('--verbose', '-v',
                         type=int, 
                         help="Ver progreso de entrenamiento.",
                         required=False,
                         default=0)
    parser.add_argument('--epochs',
                         type=int, 
                         help="Epocas de entrenamiento.",
                         required=False,
                         default=10)
    parser.add_argument('--batch_size',
                       type=int,
                       help="Cantidad de batch de entrenamiento",
                       required=False,
                       default=32)
    parser.add_argument('--sample_size',
                       type=int,
                       help="Cantidad de datos de validación",
                       required=False,
                       default=1000)
    parser.add_argument('--max_steps',
                       type=int,
                       help="Cantidad maxima de pasos.",
                       required=False,
                       default=100)
    parser.add_argument('--name_model',
                       type=str,
                       help="Nombre del modelo.",
                       required=False,
                       default='model')
    args = parser.parse_args()
    
    with open(args.dir_params, 'r') as archivo:
        # Carga el contenido en un diccionario
        params = json.load(archivo)

    params = get_params(params, args.H)

    print("Cargando datos...")
    data_Sx7 = load_data_from_json(args.dir_data)

    model = create_model(**params)

    for stack in data_Sx7:
        model.fit(np.stack(data_Sx7[stack]["States"]), 
                  np.stack(data_Sx7[stack]["Labels"]), 
                  batch_size=args.batch_size, epochs=args.epochs, validation_split=args.percentage,
                  verbose=args.verbose)
        
    validate_model(model, optimizer=GreedyV2(), data_adapter=AttentionModel(), 
                   S=args.S, H=args.H, N=(args.S * (args.H-2)), size_states=args.sample_size,
                   max_steps=args.max_steps)

    for stack in data_Sx7:
        model.fit(np.stack(data_Sx7[stack]["States"]), 
                  np.stack(data_Sx7[stack]["Labels"]), 
                  batch_size=args.batch_size, epochs=args.epochs, validation_split=args.percentage,
                  verbose=args.verbose)

    model.save(args.name_model + '.h5')

    validate_model(model, optimizer=GreedyV2(), data_adapter=AttentionModel(), 
                   S=args.S, H=args.H, N=(args.S * (args.H-2)), size_states=args.sample_size,
                   max_steps=args.max_steps)
