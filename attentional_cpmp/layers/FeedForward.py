from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
import tensorflow as tf

class FeedForward(Layer):
    """
    FeedForward Neural Network Layer

    This class implements a simple feedforward neural network layer using TensorFlow's Keras API. The network consists of multiple dense layers with dropout regularization.

    Attributes:
        dim (int): The input dimension of the network. Default value is 5 for five column models.
        activation (str): The activation function applied to each dense layer. Default is 'sigmoid'.
        dim_output (int): The output dimension of the network. Default is 5.

    Methods:
        __init__(self, dim=5, activation='sigmoid', dim_output=5)
            Initializes the FeedForward layer with specified input dimension, activation function, and output dimension.

        call(self, inputs)
            Defines the forward pass of the network.

    Usage:
        # Create a FeedForward layer
        feedforward_layer = FeedForward(dim=10, activation='relu', dim_output=3)

        # Perform a forward pass
        output = feedforward_layer(inputs)
    """
    def __init__(self, 
                 dim_input: int,  
                 dim_output: int,
                 list_neurons: list[int] = None,
                 activation: str = 'sigmoid',
                 rate: float = 0.00001,
                 n_dropout: int = 1) -> None:
        # Verificar si los parámetros requeridos tienen valores
        if dim_input is None:
            raise ValueError("dim_input has no value.")
        super(FeedForward,self).__init__()
        self.__feed = Sequential()
        
        self.__feed.add(Dense(units=dim_input, 
                              activation=activation, 
                              input_shape=((None, dim_input))))
        
        if list_neurons is not None:
            for i, num_neuron in enumerate(list_neurons[0:], start=0):
                if i % 2 == 0:
                    self.__feed.add(Dense(units=num_neuron, 
                                      activation=activation))
                else:
                    self.__feed.add(Dense(units=num_neuron, 
                                      activation="linear"))
                if n_dropout > 0:
                    if ((i+1) % n_dropout == 0) and rate > 0:
                        self.__feed.add(Dropout(rate=rate))
        
        self.__feed.add(Dense(units=dim_output, 
                              activation=activation))

    def call(self, inputs: tf.TensorArray):
        o1 = self.__d1(inputs)
        o2 = self.__d2(o1)
        d1 = self.__dp1(o2)
        o3 = self.__d3(d1)
        o4 = self.__d4(o3)
        d2 = self.__dp2(o4)
        o5 = self.__d5(d2)

        return o5
