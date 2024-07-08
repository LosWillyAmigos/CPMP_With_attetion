from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Multiply
from keras.models import Model
from keras.saving import load_model
from cpmp_ml.model.layers import FeedForward
from cpmp_ml.model.layers import DenseLayer, Model_CPMP, StackAttention
from cpmp_ml.model.layers import LayerExpandOutput
from cpmp_ml.model.layers import Concatenation
from cpmp_ml.model.layers import Reduction

def create_model(heads: int = 5,
                H: int = 5,
                optimizer: str | None = 'Adam', 
                epsilon:float=1e-6,
                num_stacks: int = 1,
                list_neuron_feed: list = [30, 45, 30, 45, 30],
                list_neuron_hide: list = None) -> Model:
    input_layer = Input(shape=(None,H+1))
    layer_attention_so = Model_CPMP(H=H,heads=heads,
                                    activation='sigmoid',
                                    epsilon=epsilon, 
                                    num_stacks=num_stacks,
                                    list_neurons_feed=list_neuron_feed,
                                    list_neuron_hide=list_neuron_hide)(input_layer)
    expand = LayerExpandOutput()(layer_attention_so)
    concatenation = Concatenation()(input_layer)
    distributed = TimeDistributed(Model_CPMP(H=H+1,heads=heads,
                                             activation='sigmoid', 
                                             epsilon=epsilon, 
                                             num_stacks=num_stacks,
                                             list_neurons_feed=list_neuron_feed, 
                                             list_neuron_hide=list_neuron_hide))(concatenation)
    unificate = Flatten()(distributed)
    mult = Multiply()([unificate,expand])
    red = Reduction()(mult)

    model = Model(inputs=input_layer,outputs=red)
    model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics= ['mae', 'mse'])

    return model

def load_cpmp_model(name: str) -> Model:
        c_o={'Model_CPMP': Model_CPMP, 
             'LayerExpandOutput': LayerExpandOutput,
             'ConcatenationLayer': Concatenation,
             'Reduction': Reduction,
             'FeedForward' : FeedForward,
             'Stack_Attention' : StackAttention,
             'DenseLayer' : DenseLayer}
        model = load_model(name, custom_objects=c_o)
        
        return model