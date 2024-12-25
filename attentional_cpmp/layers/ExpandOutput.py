from keras.api.layers import Layer
import tensorflow as tf

class ExpandOutput(Layer):
    """
    LayerExpandOutput

    This class implements a custom layer using TensorFlow's Keras API. It expands the output by repeating each element along the second axis.

    Attributes:
        None

    Methods:
        __init__(self, **kwargs)
            Initializes the LayerExpandOutput layer.

        call(self, inputs)
            Defines the forward pass of the LayerExpandOutput layer.

    Usage:
        # Create a LayerExpandOutput layer
        expand_output_layer = LayerExpandOutput()

        # Perform a forward pass
        output = expand_output_layer(inputs)
    """

    def __init__(self, **kwargs) -> None:
        super(ExpandOutput, self).__init__(**kwargs)

    
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        dim = tf.shape(inputs)[1]
        expanded = tf.repeat(inputs, repeats=dim, axis=1)

        return expanded