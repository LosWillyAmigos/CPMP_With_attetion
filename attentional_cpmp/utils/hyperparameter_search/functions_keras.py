from attentional_cpmp.layers import ModelCPMP
from attentional_cpmp.layers import ExpandOutput
from attentional_cpmp.layers import ConcatenationLayer
from attentional_cpmp.layers import Reduction
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Multiply
from keras.models import Model

from keras_tuner import HyperParameters

def build_model(hp: HyperParameters,
                max_num_neurons_layers_feed:int,
                max_num_neurons_layers_hide:int,
                max_units_neurons_feed:int,
                max_units_neurons_hide:int,
                max_key_dim:int,
                max_value_dim:int,
                max_epsilon:float,
                max_num_stacks:int,
                max_num_heads:int,
                max_n_dropout_hide:int,
                max_n_dropout_feed:int,
                loss:str,
                metrics:list,
                H:int):

    num_neurons_layers_feed = hp.Int("num_neurons_layers_feed", min_value=0, max_value=max_num_neurons_layers_feed)
    num_neurons_layers_hide = hp.Int("num_neurons_layers_hide", min_value=0, max_value=max_num_neurons_layers_hide)
    list_neurons_hide = [hp.Int(f"list_neurons_hide_{i}", min_value=1, max_value=max_units_neurons_hide) for i in range(num_neurons_layers_hide)]
    list_neurons_feed = [hp.Int(f"list_neurons_feed_{i}", min_value=1, max_value=max_units_neurons_feed) for i in range(num_neurons_layers_feed)]
    key_dim = hp.Int("value_dim", min_value=1, max_value=max_key_dim)
    value_dim = hp.Int("value_dim", min_value=0, max_value=max_value_dim)
    if value_dim == 0:
        param_val = None
    else: 
        param_val = value_dim
    epsilon = hp.Float("epsilon", min_value=1e-9, max_value=max_epsilon)
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.9)
    rate = hp.Float("rate", min_value=0.0, max_value=0.9)
    num_heads = hp.Int("num_heads", min_value=1, max_value=max_num_heads)
    num_stacks = hp.Int("num_stacks", min_value=1, max_value=max_num_stacks)
    activation_hide = hp.Choice("activation_hide", ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
    activation_feed = hp.Choice("activation_feed", ['linear', 'sigmoid', 'relu', 'softplus', 'gelu', 'elu', 'selu', 'exponential'])
    n_dropout_hide = hp.Int("n_dropout_hide", min_value=0, max_value=max_n_dropout_hide)
    n_dropout_feed = hp.Int("n_dropout_feed", min_value=0, max_value=max_n_dropout_feed)

    input_layer = Input(shape=(None,H+1))
    layer_attention_so = ModelCPMP(dim=H,
                                   list_neurons_hide=list_neurons_hide,
                                   list_neurons_feed=list_neurons_feed,
                                   key_dim=key_dim,
                                   value_dim=param_val,
                                   epsilon=epsilon,
                                   dropout=dropout,
                                   rate=rate,
                                   num_heads=num_heads,
                                   num_stacks=num_stacks,
                                   activation_hide=activation_hide,
                                   activation_feed=activation_feed,
                                   n_dropout_hide=n_dropout_hide,
                                   n_dropout_feed=n_dropout_feed)(input_layer)
    expand = ExpandOutput()(layer_attention_so)
    concatenation = ConcatenationLayer()(input_layer)
    distributed = TimeDistributed(ModelCPMP(dim=H + 1,
                                            list_neurons_hide=list_neurons_hide,
                                            list_neurons_feed=list_neurons_feed,
                                            key_dim=key_dim + 1,
                                            value_dim=param_val,
                                            epsilon=epsilon,
                                            dropout=dropout,
                                            rate=rate,
                                            num_heads=num_heads,
                                            num_stacks=num_stacks,
                                            activation_hide=activation_hide,
                                            activation_feed=activation_feed,
                                            n_dropout_hide=n_dropout_hide,
                                            n_dropout_feed=n_dropout_feed))(concatenation)
    unificate = Flatten()(distributed)
    mult = Multiply()([unificate,expand])
    red = Reduction()(mult)

    model = Model(inputs=input_layer,outputs=red)
    model.compile(optimizer='Adam', 
                  loss=loss, 
                  metrics=metrics)

    return model