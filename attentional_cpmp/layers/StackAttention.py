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
        list_neuron_hide (list[int]): List of values ​​indicating neurons in dense layers
        n_dropout (int): It indicates that every few dense layers there will be a drop.

    Methods:
        __init__(self, heads: int, dim: int, epsilon=1e-6, act='sigmoid', n_dropout=3)
            Initializes the Stack Attention layer with specified parameters.

        call(self, inputs_o, inputs_att, training=True)
            Defines the forward pass of the stack attention layer.

    Usage:
        # Create a Stack Attention layer
        stack_attention_layer = Stack_Attention(heads=8, dim=64, epsilon=1e-6, act='relu')

        # Perform a forward pass
        output = stack_attention_layer(inputs_o, inputs_att, training=True)
    """
    def __init__(self, heads: int = None,
                  dim_input: int =  None,
                  dim_output: int = None,
                  epsilon=1e-6,
                  list_neurons:list[int] = None,
                  activation: str = 'linear',
                  use_bias: bool=True,
                  kernel_initializer: str = "glorot_uniform",
                  bias_initializer: str="zeros",
                  k_l2_l1:float = 0.01,
                  b_l2_l1:float = 0.01,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  rate: float = 0.0,
                  noise_shape: Any | None = None,
                  seed: Any | None = None,
                  n_dropout: int = 1,
                  **kwargs) -> None:
        if heads is None or dim_input is None or dim_output is None: 
            raise ValueError("heads or dim has no value.")
        super(StackAttention, self).__init__()
        self.__multihead = MultiHeadAttention(num_heads=heads,key_dim=dim_input)
        self.__feed = FeedForward(dim_input=dim_input,
                                  dim_output=dim_output,
                                  list_neurons=list_neurons)
        self.__add_1 = Add()
        self.__add_2 = Add()
        self.__layer_n_1 = LayerNormalization(epsilon=epsilon)
        self.__layer_n_2 = LayerNormalization(epsilon=epsilon)
        
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs_o: tf.TensorArray, inputs_att: tf.TensorArray, training = True, **kwargs):
        att = self.__multihead(inputs_att, inputs_att, training=training)
        feed = self.__feed(att)
        add = self.__add([inputs_o,feed])
        output = self.__layer_n(add)

        return output