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
from keras.api.layers import Input
from keras.api.layers import TimeDistributed
from keras.api.layers import Reshape
from keras.api.layers import Multiply
from keras.api.models import Model
from keras.api.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from typing import Any

def create_model(H: int,
                 key_dim: Any,
                 value_dim: Any | None = None,
                 num_heads: int = 5,
                 list_neurons_hide: list[int] = None,
                 list_neurons_feed: list[int] = None,
                 dropout: float = 0,
                 rate: float = 0.5,
                 activation_hide: str = 'sigmoid',
                 activation_feed: str = 'sigmoid',
                 n_dropout_hide: int = 1,
                 n_dropout_feed: int = 1,
                 epsilon:float=1e-6,
                 num_stacks: int = 1,
                 optimizer: str | None = 'Adam',
                 loss: str = 'binary_crossentropy',
                 metrics: list[str] = ['mae', 'mse']) -> Model:
    input_layer = Input(shape=(None,H+1))
    layer_attention_so = ModelCPMP(dim=H,
                                   list_neurons_hide=list_neurons_hide,
                                   list_neurons_feed=list_neurons_feed,
                                   key_dim=key_dim,
                                   value_dim=value_dim,
                                   epsilon=epsilon,
                                   dropout=dropout,
                                   rate=rate,
                                   num_heads=num_heads,
                                   num_stacks=num_stacks,
                                   activation_hide=activation_hide,
                                   activation_feed=activation_feed,
                                   n_dropout_hide=n_dropout_hide,
                                   n_dropout_feed=n_dropout_feed)(input_layer)
    expand = ExpandOutput()(layer_attention_so)
    concatenation = ConcatenationLayer()(input_layer)
    distributed = TimeDistributed(ModelCPMP(dim=H + 1,
                                            list_neurons_hide=list_neurons_hide,
                                            list_neurons_feed=list_neurons_feed,
                                            key_dim=key_dim + 1,
                                            value_dim=value_dim,
                                            epsilon=epsilon,
                                            dropout=dropout,
                                            rate=rate,
                                            num_heads=num_heads,
                                            num_stacks=num_stacks,
                                            activation_hide=activation_hide,
                                            activation_feed=activation_feed,
                                            n_dropout_hide=n_dropout_hide,
                                            n_dropout_feed=n_dropout_feed))(concatenation)
    unificate = Reshape((-1,))(distributed)
    mult = Multiply()([unificate,expand])
    red = Reduction()(mult)

    model = Model(inputs=input_layer,outputs=red)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def plot_cpmp_model(model: Model = None, name: str = 'model', show_shapes: bool = True) -> None:
    if model is None:
        print('Model have not been initialized.')
        return
    
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
        
        print('Modelo cargado con éxito!')

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