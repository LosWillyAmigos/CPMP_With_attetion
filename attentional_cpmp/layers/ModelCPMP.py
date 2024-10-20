from keras.layers import Layer, Input
from keras.layers import Flatten
from keras.models import Model
from attentional_cpmp.layers import FeedForward
from attentional_cpmp.layers import StackAttention
import tensorflow as tf

from typing import Any

class ModelCPMP(Layer):
    """
    Model_CPMP Layer

    This class implements a custom neural network model named Model_CPMP using TensorFlow's Keras API. It combines stack attention and feedforward layers.

    Attributes:
        heads (int): The number of attention heads.
        H (int): The dimension parameter used in stack attention and feedforward layers.
        activation (str): Activation function applied to the inner layers. Default is 'sigmoid'.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.

    Methods:
        __init__(self, heads: int, H: int, activation='sigmoid', epsilon=1e-6)
            Initializes the Model_CPMP layer with specified parameters.

        call(self, input_0, training=True)
            Defines the forward pass of the Model_CPMP layer.

    Usage:
        # Create a Model_CPMP layer
        model_cpmp_layer = Model_CPMP(heads=8, H=64, activation='relu', epsilon=1e-6)

        # Perform a forward pass
        output = model_cpmp_layer(input_0, training=True)
    """
    def __init__(self, 
                 dim: int,
                 list_neurons_hide: list[int],
                 list_neurons_feed: list[int],
                 key_dim: Any,
                 value_dim: Any | None = None,
                 epsilon: float = 1e-6,
                 dropout: float = 0,
                 rate: float = 0.5,
                 num_heads: int = 3, 
                 num_stacks: int = None,
                 activation_hide: str = 'sigmoid', 
                 activation_feed: str = 'sigmoid',
                 n_dropout_hide: int = 1,
                 n_dropout_feed: int = 1,
                 **kwargs) -> None:
        super(ModelCPMP, self).__init__()
        if num_stacks is None or dim is None:
            raise ValueError("Arguments has no value.")
        self.__heads = num_heads
        self.__dim = dim + 1
        self.__num_stack_attention = num_stacks
        self.__epsilon = epsilon
        
        self.__list_neurons_hide = list_neurons_hide
        self.__list_neurons_feed = list_neurons_feed

        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__dropout = dropout
        self.__rate = rate

        self.__n_dropout_hide = n_dropout_hide
        self.__n_dropout_feed = n_dropout_feed
        self.__activation_hide = activation_hide
        self.__activation_feed = activation_feed

        inp = Input(shape=(None, dim + 1))
        attention = inp
        for _ in range(self.__num_stack_attention):
            attention = StackAttention(num_heads=self.__heads, 
                                       dim_input=self.__dim,
                                       dim_output=self.__dim,
                                       list_neurons=self.__list_neurons_hide,
                                       key_dim=self.__key_dim,
                                       value_dim=self.__value_dim,
                                       epsilon=self.__epsilon,
                                       activation_feed_hide=self.__activation_hide,
                                       n_dropout=self.__n_dropout_hide,
                                       dropout=self.__dropout,
                                       rate=self.__rate,
                                       **kwargs)(attention, attention)

        feed = FeedForward(dim_input=self.__dim,
                           dim_output=1,
                           list_neurons=self.__list_neurons_feed,
                           activation=self.__activation_feed,
                           rate=self.__rate,
                           n_dropout=self.__n_dropout_feed,
                           **kwargs)(attention)
        flttn = Flatten()(feed)

        self.__model_stacks = Model(inputs=inp, outputs=flttn)

    
    @tf.function
    def call(self, input_0: tf.TensorArray, training=True) -> None:
        out = self.__model_stacks(input_0, training=training)
        return out