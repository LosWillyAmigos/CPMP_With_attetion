from keras.regularizers import L1L2, L1, L2

from keras.layers import Layer
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from typing import Any

from attentional_cpmp.layers import FeedForward
from attentional_cpmp.layers import StackAttention
import tensorflow as tf

class ModelCPMP(Layer):
    """
    Model_CPMP Layer

    This class implements a custom neural network model named Model_CPMP using TensorFlow's Keras API. It combines stack attention and feedforward layers.

    Attributes:
        heads (int): The number of attention heads.
        H (int): The dimension parameter used in stack attention and feedforward layers.
        activation (str): Activation function applied to the inner layers. Default is 'sigmoid'.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.
        list_neurons_feed (list[int]): Number of neurons for dense layers final.
        list_neuron_hide (list[int]): Number of neurons for dense layers in stack attention.
        n_dropout (int): It indicates that every few dense layers there will be a drop.

    Methods:
        __init__(self, heads: int, H: int, activation='sigmoid', epsilon=1e-6,list_neurons_feed=None, list_neuron_hide=None)
            Initializes the Model_CPMP layer with specified parameters.

        call(self, input_0, training=True)
            Defines the forward pass of the Model_CPMP layer.

    Usage:
        # Create a Model_CPMP layer
        model_cpmp_layer = Model_CPMP(heads=8, H=64, activation='relu', epsilon=1e-6)

        # Perform a forward pass
        output = model_cpmp_layer(input_0, training=True)
    """
    def __init__(self, H: int = None,
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
                  output_shape: int = 128,
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
                  **kwargs) -> None:
        super(ModelCPMP, self).__init__()

        if num_stacks is None or H is None:
            raise ValueError("Arguments has no value.")
        if dim_input_hide is None or dim_output_hide is None:
            raise ValueError("dim_input_hide or dim_output_hide is None")
        if key_dim is None:
            raise ValueError("key_dim is None")
        if output_shape is None:
            raise ValueError("output_shape is None")

        # Build model
        inp = Input(shape=(None, H+1))
        
        ## Kernel regilarizer to overfitting
        if kernel_regularizer_dense is not None:
            if kernel_regularizer_dense == "L1":
                k_r_d = L1(l1=kernel_regularizer_dense_value)
            if kernel_regularizer_dense == "L2":
                k_r_d = L2(l2=kernel_regularizer_dense_value)
            if kernel_regularizer_dense == "L1L2":
                k_r_d = L1L2(l1=kernel_regularizer_dense_value, l2=kernel_regularizer_dense_value)
        else:
            k_r_d = None

        ## Bias regilarizer to overfitting
        if bias_regularizer_dense is not None:
            if bias_regularizer_dense == "L1":
                b_r_d = L1(l1=bias_regularizer_dense_value)
            if bias_regularizer_dense == "L2":
                b_r_d = L2(l2=bias_regularizer_dense_value)
            if bias_regularizer_dense == "L1L2":
                b_r_d = L1L2(l1=bias_regularizer_dense_value, l2=bias_regularizer_dense_value)
        else:
            b_r_d = None

        attention = Dense(units=output_shape,
                          activation=activation_dense,
                          use_bias=use_bias_dense,
                          kernel_initializer=kernel_initializer_dense,
                          bias_initializer=bias_initializer_dense,
                          kernel_regularizer=k_r_d,
                          bias_regularizer=b_r_d,
                          activity_regularizer=activity_regularizer_dense,
                          kernel_constraint=kernel_constraint_dense,
                          bias_constraint=bias_constraint_dense)(inp)

        for _ in range(num_stacks):
            attention = StackAttention(num_heads=num_heads,
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
                                       dim_input=dim_input_hide,
                                       dim_output=dim_output_hide,
                                       list_neurons=list_neurons_hide,
                                       activation_feed=activation_feed_hide,
                                       use_bias_feed=use_bias_feed_hide,
                                       kernel_initializer_feed=kernel_initializer_feed_hide,
                                       bias_initializer_feed=bias_initializer_feed_hide,
                                       kernel_regularizer_feed=kernel_regularizer_feed_hide,
                                       kernel_regularizer_feed_value=kernel_regularizer_feed_value_hide,
                                       bias_regularizer_feed=bias_regularizer_feed_hide,
                                       bias_regularizer_feed_value=bias_regularizer_feed_value_hide,
                                       activity_regularizer_feed=activity_regularizer_feed_hide,
                                       kernel_constraint_feed=kernel_constraint_feed_hide,
                                       bias_constraint_feed=bias_constraint_feed_hide,
                                       rate=rate_hide,
                                       noise_shape=noise_shape_hide,
                                       seed=seed_hide,
                                       n_dropout=n_dropout_hide,
                                       axis=axis,
                                       epsilon=epsilon,
                                       center=center,
                                       scale=scale,
                                       beta_initializer=beta_initializer,
                                       gamma_initializer=gamma_initializer,
                                       beta_regularizer=beta_regularizer,
                                       gamma_regularizer=gamma_regularizer,
                                       beta_constraint=beta_constraint,
                                       gamma_constraint=gamma_constraint)(attention, attention)

        feed = FeedForward(dim_input=output_shape, 
                           dim_output=1,
                           activation_feed=activation_output, 
                           list_neurons=list_neurons_feed_output,
                           use_bias=use_bias_output,
                           kernel_initializer_feed=kernel_initializer_output,
                           bias_initializer_feed=bias_initializer_output,
                           kernel_regularizer_feed= kernel_regularizer_feed_output,
                           kernel_regularizer_feed_value=kernel_regularizer_feed_value_output,
                           bias_regularizer_feed=bias_regularizer_feed_output,
                           bias_regularizer_feed_value=bias_regularizer_feed_value_output,
                           activity_regularizer_feed=activity_regularizer_feed_output,
                           kernel_constraint_feed=kernel_constraint_feed_output,
                           bias_constraint_feed=bias_constraint_feed_output,
                           rate=rate_output,
                           noise_shape=noise_shape_output,
                           seed=seed_output,
                           n_dropout=n_dropout_output)(attention)
        flttn = Flatten()(feed)
        
        self.__model_stacks = Model(inputs=inp, outputs=flttn)
        
    
    @tf.function 
    def call(self, input_0: tf.TensorArray, training = True, **kwargs):
        return self.__model_stacks(input_0, training=training, **kwargs)