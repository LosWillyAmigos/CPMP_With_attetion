from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras import Sequential
from keras.regularizers import l1_l2
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
                 kernel_initializer: str = "glorot_uniform",
                 bias_initializer: str="zeros",
                 k_l2_l1:float = 0.01,
                 b_l2_l1:float = 0.01,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 rate: float = 0.0,
                 noise_shape: Any | None = None,
                 seed: Any | None = None,
                 n_dropout: int = 1,
                 **kwargs) -> None:
        super(FeedForward, self).__init__()
        if dim_input is None or dim_output is None:
            raise ValueError("Input or Output is None")
        
        self.__feed = Sequential()
        
        self.__feed.add(Dense(units=dim_input, 
                              activation=activation, 
                              input_shape=((None, dim_input)),
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=l1_l2(l1=k_l2_l1, l2=k_l2_l1),
                              bias_regularizer=l1_l2(l1=b_l2_l1, l2=b_l2_l1),
                              activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint,
                              **kwargs))
        
        if list_neurons is not None:
            for i, num_neuron in enumerate(list_neurons[1:], start=0):
                if i % 2 == 0:
                    self.__feed.add(Dense(units=num_neuron, 
                                      activation=activation,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=l1_l2(l1=k_l2_l1, l2=k_l2_l1),
                                      bias_regularizer=l1_l2(l1=b_l2_l1, l2=b_l2_l1),
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint,
                                      **kwargs))
                else:
                    self.__feed.add(Dense(units=num_neuron, 
                                      activation="linear",
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=l1_l2(l1=k_l2_l1, l2=k_l2_l1),
                                      bias_regularizer=l1_l2(l1=b_l2_l1, l2=b_l2_l1),
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint,
                                      **kwargs))

                if ((i+1) % n_dropout == 0) and rate > 0:
                    self.__feed.add(Dropout(rate=rate, 
                                            noise_shape=noise_shape, 
                                            seed=seed,
                                            **kwargs))
        
        self.__feed.add(Dense(units=dim_output, 
                              activation=activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=l1_l2(l1=k_l2_l1, l2=k_l2_l1),
                              bias_regularizer=l1_l2(l1=b_l2_l1, l2=b_l2_l1),
                              activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint,
                              **kwargs))
        
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs: tf.TensorArray, training=True, **kwargs):
        return self.__feed(inputs, training=training, **kwargs)