from keras.layers import Layer
from keras.layers import Flatten
from attentional_cpmp.layers import FeedForward
from attentional_cpmp.layers import StackAttention
import tensorflow as tf

class ModelCPMP(Layer):
    """
    Model_CPMP Layer

    This class implements a custom neural network model named Model_CPMP using TensorFlow's Keras API. It combines stack attention and feedforward layers.

    Attributes:
        heads (int): The number of attention heads.
        H (int): The dimension parameter used in stack attention and feedforward layers.
        activation (str): Activation function applied to the inner layers. Default is 'sigmoid'.
        epsilon (float): Small constant for numerical stability in layer normalization. Default is 1e-6.

    Methods:
        __init__(self, heads: int, H: int, activation='sigmoid', epsilon=1e-6)
            Initializes the Model_CPMP layer with specified parameters.

        call(self, input_0, training=True)
            Defines the forward pass of the Model_CPMP layer.

    Usage:
        # Create a Model_CPMP layer
        model_cpmp_layer = Model_CPMP(heads=8, H=64, activation='relu', epsilon=1e-6)

        # Perform a forward pass
        output = model_cpmp_layer(input_0, training=True)
    """
    def __init__(self, heads: int = 3, 
                 H: int = None, 
                 num_stacks: int = None,
                 activation:str = 'sigmoid', 
                 epsilon:float=1e-6, 
                 list_neurons_feed:list=None,
                 list_neuron_hide:list=None) -> None:
        super(ModelCPMP, self).__init__()
        if num_stacks is None or H is None:
            raise ValueError("Arguments has no value.")
        self.__heads = heads
        self.__dim = H + 1
        self.__num_stack_attention = num_stacks
        self.__activation = activation
        self.__epsilon = epsilon
        self.__stack_list = []
        self.__feed = FeedForward(dim_input=self.__dim, dim_output=1, 
                                  activation='sigmoid', 
                                  list_neurons=list_neurons_feed)
        self.__flatt = Flatten()

        for _ in range(self.__num_stack_attention):
            custom_layer = StackAttention(heads=self.__heads,
                                           dim_input=self.__dim,
                                           list_neuron_hide=list_neuron_hide,
                                           epsilon=self.__epsilon,
                                           act=self.__activation)
            
            self.__stack_list.append(custom_layer)
    
    @tf.autograph.experimental.do_not_convert
    def call(self, input_0: tf.TensorArray, training=True) -> None:
        
        att = input_0

        for stack in self.__stack_list:
            att = stack(att, att, training)

        dn0 = self.__feed(att)
        fl = self.__flatt(dn0)
        return fl