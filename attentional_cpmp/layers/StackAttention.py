from attentional_cpmp.layers.FeedForward import FeedForward
from keras.layers import Layer
from keras.layers import MultiHeadAttention
from keras.layers import Add
from keras.layers import LayerNormalization
import tensorflow as tf

class StackAttention(Layer):
    """
    Stack Attention Layer

    This class implements a stack attention mechanism using TensorFlow's Keras API. It combines multi-head attention with layer normalization and dense layers.

    Attributes:
        heads (int): The number of attention heads.
        dim (int): The key dimension for multi-head attention.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.
        act (str): Activation function applied to the dense layers. Default is 'sigmoid'.

    Methods:
        __init__(self, heads: int, dim: int, epsilon=1e-6, act='sigmoid')
            Initializes the Stack Attention layer with specified parameters.

        call(self, inputs_o, inputs_att, training=True)
            Defines the forward pass of the stack attention layer.

    Usage:
        # Create a Stack Attention layer
        stack_attention_layer = Stack_Attention(heads=8, dim=64, epsilon=1e-6, act='relu')

        # Perform a forward pass
        output = stack_attention_layer(inputs_o, inputs_att, training=True)
    """
    def __init__(self, heads: int,
                  dim_input: int =  None, 
                  list_neuron_hide: list = None,
                  epsilon=1e-6, 
                  act = 'sigmoid') -> None:
        if heads is None or dim_input is None: 
            raise ValueError("heads or dim has no value.")
        super(StackAttention,self).__init__()
        self.__multihead = MultiHeadAttention(num_heads=heads,key_dim=dim_input)
        self.__feed = FeedForward(dim_input=dim_input, dim_output=dim_input, activation=act, list_neurons=list_neuron_hide)
        self.__add = Add()
        self.__layer_n = LayerNormalization(epsilon=epsilon)
    
    def call(self, inputs_o: tf.TensorArray, inputs_att: tf.TensorArray, training=True):
        att = self.__multihead(inputs_att,inputs_att, training=training)
        feed = self.__feed(att)
        add = self.__add([inputs_o,feed])
        output = self.__layer_n(add)

        return output