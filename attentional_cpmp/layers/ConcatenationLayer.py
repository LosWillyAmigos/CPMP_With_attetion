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

    
    @tf.function
    def call(self, inputs: tf.TensorArray) -> None:
        labels = tf.ones(tf.shape(inputs)[1])
        labels = tf.expand_dims(labels, axis= 0)
        labels = tf.repeat(labels, repeats= tf.shape(inputs)[0], axis= 0)

        diagonal_matrix = tf.eye(tf.shape(labels)[-1], dtype=tf.float32)

        diagonal_matrices = labels[:, :, tf.newaxis] * diagonal_matrix
        test = tf.expand_dims(diagonal_matrices, axis= -1)

        copied_matrices = tf.expand_dims(inputs, axis= 1)
        copied_matrices = tf.repeat(copied_matrices, repeats= tf.shape(labels)[1], axis= 1)

        results = Concatenate(axis= 3)([copied_matrices, test])

        return results