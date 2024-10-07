from keras.regularizers import L1L2, L1, L2

from keras.layers import Layer
from keras.layers import MultiHeadAttention
from keras.layers import Add
from keras.layers import Dense
from keras.layers import LayerNormalization
from typing import Any

import tensorflow as tf

from attentional_cpmp.layers.FeedForward import FeedForward

class StackAttention(Layer):
    """
    Stack Attention Layer

    This class implements a stack attention mechanism using TensorFlow's Keras API. It combines multi-head attention with layer normalization and dense layers.

    Attributes:
        heads (int): The number of attention heads.
        dim (int): The key dimension for multi-head attention.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.
        act (str): Activation function applied to the dense layers. Default is 'sigmoid'.

    Methods:
        __init__(self, heads: int, dim: int, epsilon=1e-6, act='sigmoid')
            Initializes the Stack Attention layer with specified parameters.

        call(self, inputs_o, inputs_att, training=True)
            Defines the forward pass of the stack attention layer.

    Usage:
        # Create a Stack Attention layer
        stack_attention_layer = Stack_Attention(heads=8, dim=64, epsilon=1e-6, act='relu')

        # Perform a forward pass
        output = stack_attention_layer(inputs_o, inputs_att, training=True)
    """
    def __init__(self, 
                  num_heads: int = None,
                  dropout : float = 0.2,
                  key_dim: int = None,
                  value_dim: int = None,
                  use_bias_multihead : bool = True,
                  output_shape: int = None,
                  attention_axes: list[int] = None,
                  kernel_initializer_multihead: str = "glorot_uniform",
                  bias_initializer_multihead: str = "zeros",
                  kernel_regularizer_multihead: Any | None = "L1L2",
                  kernel_regularizer_multihead_value: float = 0.01,
                  bias_regularizer_multihead: Any | None = "L1L2",
                  bias_regularizer_multihead_value: float = 0.01,
                  activity_regularizer_multihead: Any | None = None,
                  kernel_constraint_multihead: Any | None = None,
                  bias_constraint_multihead: Any | None = None,  
                  dim_input: int =  None,
                  dim_output: int = None,
                  list_neurons:list[int] = None,
                  activation_feed: str = 'linear',
                  use_bias_feed: bool=True,
                  kernel_initializer_feed: str = "glorot_uniform",
                  bias_initializer_feed: str="zeros",
                  kernel_regularizer_feed: Any | None = "L1L2",
                  kernel_regularizer_feed_value: float = 0.01,
                  bias_regularizer_feed: Any | None = "L1L2",
                  bias_regularizer_feed_value: float = 0.01,
                  activity_regularizer_feed: Any | None = None,
                  kernel_constraint_feed: Any | None = None,
                  bias_constraint_feed: Any | None = None,
                  rate: float = 0.2,
                  noise_shape: Any | None = None,
                  seed: Any | None = None,
                  n_dropout: int = 1,
                  axis: int = -1,
                  epsilon:float =1e-6,
                  center: bool = True,
                  scale: bool = True,
                  beta_initializer: str = "zeros",
                  gamma_initializer: str = "ones",
                  beta_regularizer: Any | None = None,
                  gamma_regularizer: Any | None = None,
                  beta_constraint: Any | None = None,
                  gamma_constraint: Any | None = None,
                  **kwargs) -> None:
        if num_heads is None or dim_input is None or dim_output is None: 
            raise ValueError("heads or dim has no value.")
        if key_dim is None:
            raise ValueError("key_dim is None")
        if output_shape is None:
            raise ValueError("output_shape is None")
        
        super(StackAttention, self).__init__()
        
        ## Kernel regilarizer to overfitting
        if kernel_regularizer_multihead is not None:
            if kernel_regularizer_multihead == "L1":
                k_r_m = L1(l1=kernel_regularizer_multihead_value)
            if kernel_regularizer_multihead == "L2":
                k_r_m = L2(l2=kernel_regularizer_multihead_value)
            if kernel_regularizer_multihead == "L1L2":
                k_r_m = L1L2(l1=kernel_regularizer_multihead_value, l2=kernel_regularizer_multihead_value)
        else:
            k_r_m = None

        ## Bias regilarizer to overfitting
        if bias_regularizer_multihead is not None:
            if bias_regularizer_multihead == "L1":
                b_r_m = L1(l1=bias_regularizer_multihead_value)
            if bias_regularizer_multihead == "L2":
                b_r_m = L2(l2=bias_regularizer_multihead_value)
            if bias_regularizer_multihead == "L1L2":
                b_r_m = L1L2(l1=bias_regularizer_multihead_value, l2=bias_regularizer_multihead_value)
        else:
            b_r_m = None

        self.__multihead = MultiHeadAttention(num_heads=num_heads,
                                              key_dim=key_dim,
                                              value_dim=value_dim,
                                              dropout=dropout,
                                              use_bias=use_bias_multihead,
                                              output_shape=output_shape,
                                              attention_axes=attention_axes,
                                              kernel_initializer=kernel_initializer_multihead,
                                              bias_initializer=bias_initializer_multihead,
                                              kernel_regularizer=k_r_m,
                                              bias_regularizer=b_r_m,
                                              activity_regularizer=activity_regularizer_multihead,
                                              kernel_constraint=kernel_constraint_multihead,
                                              bias_constraint=bias_constraint_multihead)
        
        
        self.__feed = FeedForward(dim_input=dim_input,
                                  dim_output=dim_output,
                                  list_neurons=list_neurons,
                                  activation_feed=activation_feed,
                                  use_bias_feed=use_bias_feed,
                                  kernel_initializer_feed=kernel_initializer_feed,
                                  bias_initializer_feed=bias_initializer_feed,
                                  kernel_regularizer_feed=kernel_regularizer_feed,
                                  kernel_regularizer_feed_value=kernel_regularizer_feed_value,
                                  bias_regularizer_feed=bias_regularizer_feed,
                                  bias_regularizer_feed_value=bias_regularizer_feed_value,
                                  activity_regularizer_feed=activity_regularizer_feed,
                                  kernel_constraint_feed=kernel_constraint_feed,
                                  bias_constraint_feed=bias_constraint_feed,
                                  rate=rate,
                                  noise_shape=noise_shape,
                                  seed=seed,
                                  n_dropout=n_dropout,
                                  **kwargs)
        self.__add_1 = Add()
        self.__add_2 = Add()
        self.__layer_n_1 = LayerNormalization(axis=axis,
                                              epsilon=epsilon,
                                              center=center,
                                              scale=scale,
                                              beta_initializer=beta_initializer,
                                              gamma_initializer=gamma_initializer,
                                              beta_regularizer=beta_regularizer,
                                              gamma_regularizer=gamma_regularizer,
                                              beta_constraint=beta_constraint,
                                              gamma_constraint=gamma_constraint)
        self.__layer_n_2 = LayerNormalization(axis=axis,
                                              epsilon=epsilon,
                                              center=center,
                                              scale=scale,
                                              beta_initializer=beta_initializer,
                                              gamma_initializer=gamma_initializer,
                                              beta_regularizer=beta_regularizer,
                                              gamma_regularizer=gamma_regularizer,
                                              beta_constraint=beta_constraint,
                                              gamma_constraint=gamma_constraint)
    
    @tf.function
    def call(self, inputs_o: tf.TensorArray, inputs_att: tf.TensorArray, training=True, **kwargs):
        ## Si no entrena muy bien cambiar el inputs_o por inputs_att
        att = self.__multihead(inputs_o, inputs_att, inputs_att, training=training)
        add_1 = self.__add_1([inputs_att, att])
        layer_n = self.__layer_n_1(add_1)
        feed = self.__feed(layer_n)
        add_2= self.__add_2([layer_n,feed])
        output = self.__layer_n_2(add_2)

        return output