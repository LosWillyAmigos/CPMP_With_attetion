from keras.layers import Layer
from keras.layers import Concatenate
import tensorflow as tf


class ConcatenationLayer(Layer):
    """
    ConcatenationLayer

    This class implements a custom concatenation layer using TensorFlow's Keras API. It concatenates each input tensor with a corresponding diagonal matrix to assign a stack origin probability to each matrix.

    Attributes:
        None

    Methods:
        __init__(self, **kwargs)
            Initializes the ConcatenationLayer.

        call(self, inputs)
            Defines the forward pass of the ConcatenationLayer.

    Usage:
        # Create a ConcatenationLayer
        concatenation_layer = ConcatenationLayer()

        # Perform a forward pass
        output = concatenation_layer(inputs)
    """
    def __init__(self, **kwargs) -> None:
        super(ConcatenationLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.TensorArray) -> None:
        labels = tf.ones(tf.shape(inputs)[1])
        labels = tf.expand_dims(labels, axis= 0)
        labels = tf.repeat(labels, repeats= tf.shape(inputs)[0], axis= 0)

        matriz_identidad = tf.eye(tf.shape(labels)[-1], dtype=tf.float32)

        matrices_diagonales = labels[:, :, tf.newaxis] * matriz_identidad
        test = tf.expand_dims(matrices_diagonales, axis= -1)

        matrices_copiadas = tf.expand_dims(inputs, axis= 1)
        matrices_copiadas = tf.repeat(matrices_copiadas, repeats= tf.shape(labels)[1], axis= 1)

        results = Concatenate(axis= 3)([matrices_copiadas, test])

        return results