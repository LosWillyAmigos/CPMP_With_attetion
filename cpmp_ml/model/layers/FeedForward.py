from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow import TensorArray

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
    def __init__(self, dim_input:int = None, dim_output:int = None, activation: str = 'sigmoid', list_neurons: list = None) -> None:
        # Verificar si los parámetros requeridos tienen valores
        super(FeedForward,self).__init__()
        self.__dense_input = Dense(dim_input, activation=activation)
        if list_neurons is not None:
            self.__dense_list = [Dense(neurons, activation=activation) for neurons in list_neurons]
        else:
            self.__dense_list = []
        self.__dense_output = Dense(dim_output, activation=activation)
    
    def call(self, inputs: TensorArray):
        out = self.__dense_input(inputs)

        for layer in self.__dense_list:
            out = layer(out)

        out = self.__dense_output(out)
        return out