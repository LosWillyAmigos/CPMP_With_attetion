from keras.layers import Layer
import tensorflow as tf

class Reduction(Layer):
    """
    Reduction Layer

    This class implements a custom reduction layer using TensorFlow's Keras API. It reduces the input tensor by removing the diagonal elements.

    Attributes:
        None

    Methods:
        __init__(self)
            Initializes the Reduction layer.

        call(self, arr)
            Defines the forward pass of the Reduction layer.

    Usage:
        # Create a Reduction layer
        reduction_layer = Reduction()

        # Perform a forward pass
        output = reduction_layer(arr)
    """
    def __init__(self) -> None:
        super(Reduction, self).__init__(trainable=False)

    def call(self, arr: tf.Tensor) -> tf.Tensor:
        S = tf.sqrt(tf.cast(tf.shape(arr)[1], dtype=tf.float32))

        aux = tf.math.logical_not(tf.eye(S, dtype=tf.bool))
        mask = tf.reshape(aux, [-1])
        
        output = tf.boolean_mask(arr, mask, axis=1)

        return output