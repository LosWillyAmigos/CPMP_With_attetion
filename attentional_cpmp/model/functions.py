from attentional_cpmp.layers import ModelCPMP
from attentional_cpmp.layers import ExpandOutput
from attentional_cpmp.layers import ConcatenationLayer
from attentional_cpmp.layers import Reduction
from attentional_cpmp.layers import FeedForward
from attentional_cpmp.layers import StackAttention

from cpmp_ml.generators import generate_data_v3
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.validations import validate_model

from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Multiply
from keras.models import Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def create_model(heads: int = 5,
                H: int = 5,
                optimizer: str | None = 'Adam', 
                epsilon:float=1e-6,
                num_stacks: int = 1,
                list_neuron_feed: list = [30, 45, 30, 45, 30],
                list_neuron_hide: list = None,
                n_dropout: int = 3,
                dropout: float = 0.5) -> Model:
    input_layer = Input(shape=(None,H+1))
    layer_attention_so = ModelCPMP(H=H,heads=heads,
                                    activation='sigmoid',
                                    epsilon=epsilon, 
                                    num_stacks=num_stacks,
                                    list_neurons_feed=list_neuron_feed,
                                    list_neuron_hide=list_neuron_hide,
                                    n_dropout=n_dropout,
                                    dropout=dropout)(input_layer)
    expand = ExpandOutput()(layer_attention_so)
    concatenation = ConcatenationLayer()(input_layer)
    distributed = TimeDistributed(ModelCPMP(H=H+1,heads=heads,
                                             activation='sigmoid', 
                                             epsilon=epsilon, 
                                             num_stacks=num_stacks,
                                             list_neurons_feed=list_neuron_feed, 
                                             list_neuron_hide=list_neuron_hide,
                                             n_dropout=n_dropout,
                                             dropout=dropout))(concatenation)
    unificate = Flatten()(distributed)
    mult = Multiply()([unificate,expand])
    red = Reduction()(mult)

    model = Model(inputs=input_layer,outputs=red)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics= ['mae', 'mse'])

    return model

def plot_cpmp_model(model: Model = None, name: str = 'model', show_shapes: bool = True) -> None:
    if model is None:
        raise ValueError('Model have not been initialized.')
    
    input_shape = model.layers[0].input_shape[0]
    name_img = name + 'x' + str(input_shape[2] - 1) + '.png'

    plot_model(model, show_shapes=show_shapes, to_file= name_img)

    imagen = plt.imread(name_img)
    plt.figure(figsize=(20, 25))
    plt.imshow(imagen)
    plt.axis('off')
    plt.show()

def load_cpmp_model(name: str) -> Model:
        c_o={'ModelCPMP': ModelCPMP, 
             'ExpandOutput': ExpandOutput,
             'ConcatenationLayer': ConcatenationLayer,
             'Reduction': Reduction,
             'FeedForward' : FeedForward,
             'StackAttention' : StackAttention}
        model = tf.keras.models.load_model(name,custom_objects=c_o)
        
        return model

def reinforcement_training(model: Model, S: int, H: int, N: int, validate_optimizer: OptimizerStrategy, adapter: DataAdapter,
                           sample_size: int = 50000, iter: int = 5, max_steps: int = 30, epochs: int = 5, 
                           batch_size: int = 20, verbose: bool = True, perms_by_layout: int = 1) -> None:
    optimizer = GreedyModel(model, adapter)

    for i in range(iter):
        if verbose: print(f"Step {i + 1}")

        data, labels = generate_data_v3(optimizer, adapter, S, H, N, sample_size, batch_size, 
                                        perms_by_layout= perms_by_layout, max_steps= max_steps)

        data = np.stack(data)
        labels = np.stack(labels)

        model.fit(data, labels, epochs= epochs, verbose= verbose)
        results_model, results_greedy = validate_model(model, validate_optimizer, adapter, S, H, N, 1000, max_steps= max_steps)

        del data, labels

        if verbose: print('')
        if results_model > 96.0: break