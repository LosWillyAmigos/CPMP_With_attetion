from keras.layers import Layer
from keras.layers import Concatenate, Flatten, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Input
import tensorflow as tf
import numpy as np

class Model_CPMP(Layer):
    def __init__(self, num_layer_attention_add: int = 1,
                   heads: int = 5, S: int = 5, H: int = 5, ):
        super(Model_CPMP,self).__init__()
        self.__num_layer_attention_add = num_layer_attention_add
        self.__heads = heads
        self.__S = S
        self.__H = H

    def call(self, input):
        reshape = self.__stack_attention(self.__heads, self.__H + 1, input, input)
        for i in range(self.__num_layer_attention_add):
            reshape = self.__stack_attention(self.__heads, self.__H + 1, reshape, input)

        reshape = Flatten()(reshape)
        hidden1 = Dense(self.__H * 6, activation='sigmoid')(reshape)
        dropout_1 = Dropout(0.5)(hidden1)
        hidden2 = Dense(self.__H * 6, activation='sigmoid')(dropout_1)
        output = Dense(self.__S, activation='sigmoid')(hidden2)

        return output
    
    def __feed_forward_layer(self, input: None, num_neurons: int) -> Dense:
        # capa de feed para que el modelo pueda aprender
        layer = Dense(num_neurons, activation='sigmoid')(input)
        layer = Dense(num_neurons)(layer)
        return layer
    
    def __attention_layer(self, heads: int, d_model: int, reshape: None) -> MultiHeadAttention:
        attention = MultiHeadAttention(num_heads=heads, key_dim=d_model)(reshape, reshape)
        return attention
    
    def __normalization_layer(self, attention: None, input: None) -> LayerNormalization:
        layer = Add()([input, attention])
        layer = LayerNormalization(epsilon=1e-6)(layer)

        return layer
    def __stack_attention(self, heads: int, d_model: int, reshape: None, input: None) -> Dense:
        # por si se debe modificar la dimensión
        attention = self.__attention_layer(heads, d_model, reshape)
        normalization = self.__normalization_layer(input, attention)
        feed = self.__feed_forward_layer(normalization, d_model)

        return feed


class ConcatenationLayer(Layer):
    def __init__(self, **kwargs):
        super(ConcatenationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        matrix, labels = inputs

        # Crear una matriz identidad de la misma forma que los arreglos
        matriz_identidad = tf.eye(labels.shape[-1], dtype=tf.float32)

        # Multiplicar cada arreglo por la matriz identidad para obtener la matriz diagonal
        matrices_diagonales = labels[:, :, tf.newaxis] * matriz_identidad

        test = tf.expand_dims(matrices_diagonales, axis= -1)

        matrices_copiadas = tf.expand_dims(matrix, axis= 1)
        matrices_copiadas = tf.repeat(matrices_copiadas, repeats= labels.shape[1], axis= 1)

        results = Concatenate(axis= 3)([matrices_copiadas, test])

        return results
    
class LayerExpandOutput(Layer):
    def __init__(self, **kwargs):
        super(LayerExpandOutput, self).__init__(**kwargs, trainable=False)

    def call(self, inputs):
        dim = tf.shape(inputs)[1]
        expanded = tf.repeat(inputs, repeats= dim, axis= 1)

        return expanded
class OutputMultiplication(Layer):
    def __init__(self) -> None:
        super(OutputMultiplication,self).__init__(trainable=False)

    def call(self, arr1: tf.TensorArray, arr2: tf.TensorArray) -> tf.TensorArray:
        return arr1 * arr2