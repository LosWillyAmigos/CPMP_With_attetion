from keras.regularizers import L1L2, L1, L2

from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras import Sequential
from typing import Any

import tensorflow as tf

class FeedForward(Layer):
    """
    FeedForward Neural Network Layer

    This class implements a simple feedforward neural network layer using TensorFlow's Keras API. The network consists of multiple dense layers with dropout regularization.

    Attributes:
        
    Methods:
        __init__()

        call(self, inputs, training=True, **kwargs)
            Defines the forward pass of the network.

    Usage:
    
    """
    def __init__(self, 
                 dim_input: int = None, 
                 dim_output: int = None, 
                 activation: str = 'linear',
                 list_neurons: list[int] = None,
                 use_bias: bool=True,
                 kernel_initializer_feed: str = "glorot_uniform",
                 bias_initializer_feed: str="zeros",
                 kernel_regularizer_feed: Any | None = "L1L2",
                 kernel_regularizer_feed_value: float = 0.01,
                 bias_regularizer_feed: Any | None = "L1L2",
                 bias_regularizer_feed_value: float = 0.01,
                 activity_regularizer_feed: Any | None = None,
                 kernel_constraint_feed: Any | None = None,
                 bias_constraint_feed: Any | None = None,
                 rate: float = 0.000001,
                 noise_shape: Any | None = None,
                 seed: Any | None = None,
                 n_dropout: int = 1,
                 **kwargs) -> None:
        super(FeedForward, self).__init__()
        if dim_input is None or dim_output is None:
            raise ValueError("Input or Output is None")
        
        ## Kernel regilarizer to overfitting
        if kernel_regularizer_feed is not None:
            if kernel_regularizer_feed== "L1":
                k_r_f = L1(l1=kernel_regularizer_feed_value)
            if kernel_regularizer_feed == "L2":
                k_r_f = L2(l2=kernel_regularizer_feed_value)
            if kernel_regularizer_feed == "L1L2":
                k_r_f = L1L2(l1=kernel_regularizer_feed_value, l2=kernel_regularizer_feed_value)
        else:
            k_r_f = None

        ## Bias regilarizer to overfitting
        if bias_regularizer_feed is not None:
            if bias_regularizer_feed == "L1":
                b_r_f = L1(l1=bias_regularizer_feed_value)
            if bias_regularizer_feed == "L2":
                b_r_f = L2(l2=bias_regularizer_feed_value)
            if bias_regularizer_feed == "L1L2":
                b_r_f = L1L2(l1=bias_regularizer_feed_value, l2=bias_regularizer_feed_value)
        else:
            b_r_f = None
        self.__feed = Sequential()
        
        self.__feed.add(Dense(units=dim_input, 
                              activation=activation, 
                              input_shape=((None, dim_input)),
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer_feed,
                              bias_initializer=bias_initializer_feed,
                              kernel_regularizer=k_r_f,
                              bias_regularizer=b_r_f,
                              activity_regularizer=activity_regularizer_feed,
                              kernel_constraint=kernel_constraint_feed,
                              bias_constraint=bias_constraint_feed))
        
        if list_neurons is not None:
            for i, num_neuron in enumerate(list_neurons[0:], start=0):
                if i % 2 == 0:
                    self.__feed.add(Dense(units=num_neuron, 
                                      activation=activation,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer_feed,
                                      bias_initializer=bias_initializer_feed,
                                      kernel_regularizer=k_r_f,
                                      bias_regularizer=b_r_f,
                                      activity_regularizer=activity_regularizer_feed,
                                      kernel_constraint=kernel_constraint_feed,
                                      bias_constraint=bias_constraint_feed))
                else:
                    self.__feed.add(Dense(units=num_neuron, 
                                      activation="linear",
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer_feed,
                                      bias_initializer=bias_initializer_feed,
                                      kernel_regularizer=k_r_f,
                                      bias_regularizer=b_r_f,
                                      activity_regularizer=activity_regularizer_feed,
                                      kernel_constraint=kernel_constraint_feed,
                                      bias_constraint=bias_constraint_feed))

                if ((i+1) % n_dropout == 0) and rate > 0:
                    self.__feed.add(Dropout(rate=rate, 
                                            noise_shape=noise_shape, 
                                            seed=seed))
        
        self.__feed.add(Dense(units=dim_output, 
                              activation=activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer_feed,
                              bias_initializer=bias_initializer_feed,
                              kernel_regularizer=k_r_f,
                              bias_regularizer=b_r_f,
                              activity_regularizer=activity_regularizer_feed,
                              kernel_constraint=kernel_constraint_feed,
                              bias_constraint=bias_constraint_feed))
    
    
    @tf.function
    def call(self, inputs: tf.TensorArray, training=True, **kwargs):
        return self.__feed(inputs, training=training, **kwargs)
