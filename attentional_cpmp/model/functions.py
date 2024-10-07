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
from typing import Any

def create_model(H: int = None,
                  num_stacks: int = None,
                  num_heads: int = None,
                  activation_dense: Any | None = "linear",
                  use_bias_dense: bool = True,
                  kernel_initializer_dense: str = "glorot_uniform",
                  bias_initializer_dense: str = "zeros",
                  kernel_regularizer_dense: Any | None = "L1L2",
                  kernel_regularizer_dense_value: float = 0.01,
                  bias_regularizer_dense: Any | None = "L1L2",
                  bias_regularizer_dense_value: float = 0.01,
                  activity_regularizer_dense: Any | None = None,
                  kernel_constraint_dense: Any | None = None,
                  bias_constraint_dense: Any | None = None,
                  dropout : float = 0.2,
                  key_dim: int = None,
                  value_dim: int = None,
                  use_bias_multihead : bool = True,
                  output_shape: int = None,
                  attention_axes: list[int] = None,
                  kernel_initializer_multihead: str = "glorot_uniform",
                  bias_initializer_multihead: str = "zeros",
                  kernel_regularizer_multihead: str = "L1L2",
                  kernel_regularizer_multihead_value: float = 0.01,
                  bias_regularizer_multihead: str = "L1L2",
                  bias_regularizer_multihead_value: float = 0.01,
                  activity_regularizer_multihead: Any | None = None,
                  kernel_constraint_multihead: Any | None = None,
                  bias_constraint_multihead: Any | None = None,  
                  dim_input_hide: int =  None,
                  dim_output_hide: int = None,
                  list_neurons_hide:list[int] = None,
                  activation_feed_hide: str = 'linear',
                  use_bias_feed_hide: bool=True,
                  kernel_initializer_feed_hide: str = "glorot_uniform",
                  bias_initializer_feed_hide: str ="zeros",
                  kernel_regularizer_feed_hide: Any | None = "L1L2",
                  kernel_regularizer_feed_value_hide: float = 0.01,
                  bias_regularizer_feed_hide: Any | None = "L1L2",
                  bias_regularizer_feed_value_hide: float = 0.01,
                  activity_regularizer_feed_hide: Any | None = None,
                  kernel_constraint_feed_hide: Any | None = None,
                  bias_constraint_feed_hide: Any | None = None,
                  rate_hide: float = 0.2,
                  noise_shape_hide: Any | None = None,
                  seed_hide: Any | None = None,
                  n_dropout_hide: int = 1,
                  axis: int = -1,
                  epsilon:float = 1e-6,
                  center: bool = True,
                  scale: bool = True,
                  beta_initializer: str = "zeros",
                  gamma_initializer: str = "ones",
                  beta_regularizer: Any | None = None,
                  gamma_regularizer: Any | None = None,
                  beta_constraint: Any | None = None,
                  gamma_constraint: Any | None = None,
                  activation_output: Any | None = "sigmoid",
                  list_neurons_feed_output: list[int] = None,
                  use_bias_output: bool = True,
                  kernel_initializer_output: str = "glorot_uniform",
                  bias_initializer_output: str = "zeros",
                  kernel_regularizer_feed_output: Any | None = "L1L2",
                  kernel_regularizer_feed_value_output: float = 0.01,
                  bias_regularizer_feed_output: Any | None = "L1L2",
                  bias_regularizer_feed_value_output: float = 0.01,
                  activity_regularizer_feed_output: Any | None = None,
                  kernel_constraint_feed_output: Any | None = None,
                  bias_constraint_feed_output: Any | None = None,
                  rate_output: float = 0.2,
                  noise_shape_output: Any | None = None,
                  seed_output: Any | None = None,
                  n_dropout_output: int = 1,
                  optimizer: Any | None = 'Adam',
                  loss: Any | None = "binary_crossentropy",
                  metrics: Any | None = ['mae', 'mse'],
                  **kwargs) -> Model:
    
    input_layer = Input(shape=(None,H+1))

    layer_attention_so = ModelCPMP(H=H, num_heads=num_heads,
                                   num_stacks=num_stacks,
                                   activation_dense=activation_dense,
                                   use_bias_dense=use_bias_dense,
                                   kernel_initializer_dense=kernel_initializer_dense,
                                   bias_initializer_dense=bias_initializer_dense,
                                   kernel_regularizer_dense=kernel_regularizer_dense,
                                   kernel_regularizer_dense_value=kernel_regularizer_dense_value,
                                   bias_regularizer_dense=bias_regularizer_dense,
                                   bias_regularizer_dense_value=bias_regularizer_dense_value,
                                   activity_regularizer_dense=activity_regularizer_dense,
                                   kernel_constraint_dense=kernel_constraint_dense,
                                   bias_constraint_dense=bias_constraint_dense,
                                   dropout=dropout,
                                   key_dim=key_dim,
                                   value_dim=value_dim,
                                   use_bias_multihead=use_bias_multihead,
                                   output_shape=output_shape,
                                   attention_axes=attention_axes,
                                   kernel_initializer_multihead=kernel_initializer_multihead,
                                   bias_initializer_multihead=bias_initializer_multihead,
                                   kernel_regularizer_multihead=kernel_regularizer_multihead,
                                   kernel_regularizer_multihead_value=kernel_regularizer_multihead_value,
                                   bias_regularizer_multihead=bias_regularizer_multihead,
                                   bias_regularizer_multihead_value=bias_regularizer_multihead_value,
                                   activity_regularizer_multihead=activity_regularizer_multihead,
                                   kernel_constraint_multihead=kernel_constraint_multihead,
                                   bias_constraint_multihead=bias_constraint_multihead,
                                   dim_input_hide=dim_input_hide,
                                   dim_output_hide=dim_output_hide,
                                   list_neurons_hide=list_neurons_hide,
                                   activation_feed_hide=activation_feed_hide,
                                   use_bias_feed_hide=use_bias_feed_hide,
                                   kernel_initializer_feed_hide=kernel_initializer_feed_hide,
                                   bias_initializer_feed_hide=bias_initializer_feed_hide,
                                   kernel_regularizer_feed_hide=kernel_regularizer_feed_hide,
                                   kernel_regularizer_feed_value_hide=kernel_regularizer_feed_value_hide,
                                   bias_regularizer_feed_hide=bias_regularizer_feed_hide,
                                   bias_regularizer_feed_value_hide=bias_regularizer_feed_value_hide,
                                   activity_regularizer_feed_hide=activity_regularizer_feed_hide,
                                   kernel_constraint_feed_hide=kernel_constraint_feed_hide,
                                   bias_constraint_feed_hide=bias_constraint_feed_hide,
                                   rate_hide=rate_hide,
                                   noise_shape_hide=noise_shape_hide,
                                   seed_hide=seed_hide,
                                   n_dropout_hide=n_dropout_hide,
                                   axis=axis,
                                   epsilon=epsilon,
                                   center=center,
                                   scale=scale,
                                   beta_initializer=beta_initializer,
                                   gamma_initializer=gamma_initializer,
                                   beta_regularizer=beta_regularizer,
                                   gamma_regularizer=gamma_regularizer,
                                   beta_constraint=beta_constraint,
                                   gamma_constraint=gamma_constraint,
                                   activation_output=activation_output,
                                   list_neurons_feed_output=list_neurons_feed_output,
                                   use_bias_output=use_bias_output,
                                   kernel_initializer_output=kernel_initializer_output,
                                   bias_initializer_output=bias_initializer_output,
                                   kernel_regularizer_feed_output=kernel_regularizer_feed_output,
                                   kernel_regularizer_feed_value_output=kernel_regularizer_feed_value_output,
                                   bias_regularizer_feed_output=bias_regularizer_feed_output,
                                   bias_regularizer_feed_value_output=bias_regularizer_feed_value_output,
                                   activity_regularizer_feed_output=activity_regularizer_feed_output,
                                   kernel_constraint_feed_output=kernel_constraint_feed_output,
                                   bias_constraint_feed_output=bias_constraint_feed_output,
                                   rate_output=rate_output,
                                   noise_shape_output=noise_shape_output,
                                   seed_output=seed_output,
                                   n_dropout_output=n_dropout_output)(input_layer)
    
    expand = ExpandOutput()(layer_attention_so)
    concatenation = ConcatenationLayer()(input_layer)

    distributed = TimeDistributed(ModelCPMP(H=H+1, num_heads=num_heads,
                                   num_stacks=num_stacks,
                                   activation_dense=activation_dense,
                                   use_bias_dense=use_bias_dense,
                                   kernel_initializer_dense=kernel_initializer_dense,
                                   bias_initializer_dense=bias_initializer_dense,
                                   kernel_regularizer_dense=kernel_regularizer_dense,
                                   kernel_regularizer_dense_value=kernel_regularizer_dense_value,
                                   bias_regularizer_dense=bias_regularizer_dense,
                                   bias_regularizer_dense_value=bias_regularizer_dense_value,
                                   activity_regularizer_dense=activity_regularizer_dense,
                                   kernel_constraint_dense=kernel_constraint_dense,
                                   bias_constraint_dense=bias_constraint_dense,
                                   dropout=dropout,
                                   key_dim=key_dim,
                                   value_dim=value_dim,
                                   use_bias_multihead=use_bias_multihead,
                                   output_shape=output_shape,
                                   attention_axes=attention_axes,
                                   kernel_initializer_multihead=kernel_initializer_multihead,
                                   bias_initializer_multihead=bias_initializer_multihead,
                                   kernel_regularizer_multihead=kernel_regularizer_multihead,
                                   kernel_regularizer_multihead_value=kernel_regularizer_multihead_value,
                                   bias_regularizer_multihead=bias_regularizer_multihead,
                                   bias_regularizer_multihead_value=bias_regularizer_multihead_value,
                                   activity_regularizer_multihead=activity_regularizer_multihead,
                                   kernel_constraint_multihead=kernel_constraint_multihead,
                                   bias_constraint_multihead=bias_constraint_multihead,
                                   dim_input_hide=dim_input_hide,
                                   dim_output_hide=dim_output_hide,
                                   list_neurons_hide=list_neurons_hide,
                                   activation_feed_hide=activation_feed_hide,
                                   use_bias_feed_hide=use_bias_feed_hide,
                                   kernel_initializer_feed_hide=kernel_initializer_feed_hide,
                                   bias_initializer_feed_hide=bias_initializer_feed_hide,
                                   kernel_regularizer_feed_hide=kernel_regularizer_feed_hide,
                                   kernel_regularizer_feed_value_hide=kernel_regularizer_feed_value_hide,
                                   bias_regularizer_feed_hide=bias_regularizer_feed_hide,
                                   bias_regularizer_feed_value_hide=bias_regularizer_feed_value_hide,
                                   activity_regularizer_feed_hide=activity_regularizer_feed_hide,
                                   kernel_constraint_feed_hide=kernel_constraint_feed_hide,
                                   bias_constraint_feed_hide=bias_constraint_feed_hide,
                                   rate_hide=rate_hide,
                                   noise_shape_hide=noise_shape_hide,
                                   seed_hide=seed_hide,
                                   n_dropout_hide=n_dropout_hide,
                                   axis=axis,
                                   epsilon=epsilon,
                                   center=center,
                                   scale=scale,
                                   beta_initializer=beta_initializer,
                                   gamma_initializer=gamma_initializer,
                                   beta_regularizer=beta_regularizer,
                                   gamma_regularizer=gamma_regularizer,
                                   beta_constraint=beta_constraint,
                                   gamma_constraint=gamma_constraint,
                                   activation_output=activation_output,
                                   list_neurons_feed_output=list_neurons_feed_output,
                                   use_bias_output=use_bias_output,
                                   kernel_initializer_output=kernel_initializer_output,
                                   bias_initializer_output=bias_initializer_output,
                                   kernel_regularizer_feed_output=kernel_regularizer_feed_output,
                                   kernel_regularizer_feed_value_output=kernel_regularizer_feed_value_output,
                                   bias_regularizer_feed_output=bias_regularizer_feed_output,
                                   bias_regularizer_feed_value_output=bias_regularizer_feed_value_output,
                                   activity_regularizer_feed_output=activity_regularizer_feed_output,
                                   kernel_constraint_feed_output=kernel_constraint_feed_output,
                                   bias_constraint_feed_output=bias_constraint_feed_output,
                                   rate_output=rate_output,
                                   noise_shape_output=noise_shape_output,
                                   seed_output=seed_output,
                                   n_dropout_output=n_dropout_output))(concatenation)
    
    unificate = Flatten()(distributed)
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