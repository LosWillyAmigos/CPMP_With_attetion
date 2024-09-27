from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
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
    def __init__(self, dim_input: int, activation: str = 'sigmoid', dim_output: int = 1) -> None:
        # Verificar si los parámetros requeridos tienen valores
        if dim_input is None:
            raise ValueError("dim_input has no value.")
        super(FeedForward,self).__init__()
        self.__d1 = Dense(dim_input, activation='linear')
        self.__d2 = Dense(dim_input * 4, activation='sigmoid')
        self.__dp1 = Dropout(0.2)
        self.__d3 = Dense(dim_input * 3, activation='sigmoid')
        self.__dp2 = Dropout(0.2)
        self.__d4 = Dense(dim_input * 2, activation='sigmoid')
        self.__d5 = Dense(dim_output, activation=activation)

    def call(self, inputs: tf.TensorArray):
        o1 = self.__d1(inputs)
        o2 = self.__d2(o1)
        d1 = self.__dp1(o2)
        o3 = self.__d3(d1)
        o4 = self.__d4(o3)
        d2 = self.__dp2(o4)
        o5 = self.__d5(d2)

        return o5
