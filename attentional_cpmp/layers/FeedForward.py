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
        list_neurons (list[int]): Number of neurons for dense layers.
        n_dropout (int): It indicates that every few dense layers there will be a drop.

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
                 dim_input: int = None, 
                 dim_output: int = None, 
                 activation: str = 'sigmoid', 
                 list_neurons: list = None, 
                 n_dropout: int = 3) -> None:
        super(FeedForward, self).__init__()
        if dim_input is None or dim_output is None:
            raise ValueError("Input or Output is None")
        
        self.__dense_input = Dense(dim_input, activation=activation)
        
        if list_neurons is not None:
            self.__dense_list = []
            contador = 1
            total_layers = len(list_neurons)
            
            for index in range(total_layers):
                # Añadir la capa densa
                self.__dense_list.append(Dense(list_neurons[index], activation=activation))
                
                # Añadir la capa dropout si no es una de las últimas tres capas
                if index < total_layers:
                    if contador == n_dropout:
                        self.__dense_list.append(Dropout(0.5))
                        contador = 1
                        continue
                    contador += 1
        
        else:
            self.__dense_list = []
        
        self.__dense_output = Dense(dim_output, activation=activation)

    def call(self, inputs: tf.TensorArray):
        out = self.__dense_input(inputs)

        for layer in self.__dense_list:
            out = layer(out)

        out = self.__dense_output(out)
        return out