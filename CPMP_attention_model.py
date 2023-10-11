import keras
import numpy as np
from keras.layers import Input, Add, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from keras.layers import TimeDistributed, Concatenate, Multiply
from keras.models import Model, load_model
from Layers import ConcatenationLayer, LayerExpandOutput, OutputMultiplication, Model_CPMP
import matplotlib.pyplot as plt
import tensorflow as tf

#*************** | create_model() | ****************#
# El proposito de esta función es generar el modelo #
# para resolver el problema CPMP con ayuda de capas #
# de atención, una capa Flatten, una Dropout y      #
# capaz Dense.                                      #
#                                                   #
# Input:                                            #
#     - heads: número de cabezales que se usarán    #
#              en la capa de atención.              #
#     - S: Cantidad máxima de stacks que aceptará   #
#          el modelo.                               #
#     - H: Altura máxima de los stacks que          #
#          aceptará el modelo.                      #
#     - optimizer: Optimizador que usará el         #
#                  modelo a la hora del             #
#                  entrenamiento.                   #
# Output:                                           #
#     Retorna el modelo capaz de resolver el        #
#     problema CPMP.                                #
"""def create_model(heads, S, H , optimizer):
    input = Input(shape= (S, H+1))

    reshape = stack_attention(heads, H + 1, input, input)
    reshape = stack_attention(heads, H + 1, reshape, input)

    reshape = Flatten()(reshape)
    hidden1 = Dense(H * 6, activation='sigmoid')(reshape)
    dropout_1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(H * 6, activation='sigmoid')(dropout_1)
    output = Dense(S,activation='softmax')(hidden2)

    model = Model(inputs=input,outputs=output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics= ['mae', 'mse', 'accuracy'])

    return model"""

class CPMP_attention_model():
    def __init__(self) -> None:
        self.__S = 0
        self.__H = 0
        self.__model = None

    #********* | __normalization_layer() | **********#
    # El proposito de esta función es crear una      #
    # capa que añada la salida de una capa de        #
    # atención con el input realizado en el modelo   #
    # para disminuir la perdida de memoria y por     #
    # último aplicar una capa de normalización.      # 
    #                                                #
    # Input:                                         #
    #     - input: Entrada que representa la         #
    #              información a la cual se          #
    #              le desea hacer incapie.           #
    #     - attention: Output realizado por una      #
    #                  capa de atención.             #
    #                                                #
    # Output:                                        #
    #      Retorna la unión entre una capa de        #
    #      normalización con la capa de unión.       #  
    def __normalization_layer(self, attention: None, input: None) -> LayerNormalization:
        layer = Add()([input, attention])
        layer = LayerNormalization(epsilon=1e-6)(layer)

        return layer

    #************* | __feed_forward_layer() | **************#
    # El proposito de esta función es generar una capa de   #
    # feed para que el modelo pueda aprender de lo          #
    # realizado en previamente a través de capas densas.    #
    #                                                       #
    # Input:                                                #
    #     - input: Capa a la cual irá conectada.            #
    #     - num_neuron: Número de neuronas.                 #
    # Output:                                               #
    #     Retorna las conexiones de 2 capas densas con      #
    #     el número de neuronas asignadas.                  #
    def __feed_forward_layer(self, input: None, num_neurons: int) -> Dense:
        # capa de feed para que el modelo pueda aprender
        layer = Dense(num_neurons, activation='sigmoid')(input)
        layer = Dense(num_neurons)(layer)
        return layer
    
    #****************** | __attention_layer() | *******************#
    # El proposito de esta funicón es generar una capa             #
    # MultiHeadAttention con un cierto número de cabezales y       #
    # con las conexiones entregadas.                               #
    #                                                              #
    # Input:                                                       #
    #     - heads: Número de cabezales.                            #
    #     - d_model: Dimensión de cada entrada dentro de la        #
    #                capa.                                         #
    #     - reshape: La conexión a una capa adyacente.             #
    # Output:                                                      #
    #     Retorna la conexión a una capa MultiHeadAttention.       #
    def __attention_layer(self, heads: int, d_model: int, reshape: None) -> MultiHeadAttention:
        attention = MultiHeadAttention(num_heads=heads, key_dim=d_model)(reshape, reshape)
        return attention

    #************** | __stack_attention() | **************#
    # El proposito de esta función es generar un stack    #
    # de capaz de atención.                               #
    #                                                     #
    # Input:                                              #
    #     - heads: Número de cabezales dentro de la capa  #
    #              de atención.                           #
    #     - d_model: Dimensión de la entrada a la capa    #
    #                de atención.                         #
    #     - reshape: Capa de reshape realizado al input.  #
    #     - input: Datos ingresados al modelo para        #
    #              diminuir la perdida de memoria.        #
    # Output:                                             #
    #     Retorna todas las capas conectadas.             #
    def __stack_attention(self, heads: int, d_model: int, reshape: None, input: None) -> Dense:
        # por si se debe modificar la dimensión
        attention = self.__attention_layer(heads, d_model, reshape)
        normalization = self.__normalization_layer(input, attention)
        feed = self.__feed_forward_layer(normalization, d_model)

        return feed
    
    def __model_so(self, input: None, num_layer_attention_add: int = 1,
                   heads: int = 5, S: int = 5, H: int = 5, 
                   ) -> Dense:
        reshape = self.__stack_attention(heads, H + 1, input, input)
        for i in range(num_layer_attention_add):
            reshape = self.__stack_attention(heads, H + 1, reshape, input)

        reshape = Flatten()(reshape)
        hidden1 = Dense(H * 6, activation='sigmoid')(reshape)
        dropout_1 = Dropout(0.5)(hidden1)
        hidden2 = Dense(H * 6, activation='sigmoid')(dropout_1)
        output = Dense(S, activation='sigmoid')(hidden2)

        return output

    def __model_sd(self, num_layer_attention_add: int = 1,
                   heads: int = 5, S: int = 5, H: int = 5, 
                   ) -> Dense:
        input = Input(shape=(S, H + 2))

        reshape = self.__stack_attention(heads, H + 2, input, input)
        for i in range(num_layer_attention_add):
            reshape = self.__stack_attention(heads, H + 2, reshape, input)

        reshape = Flatten()(reshape)
        hidden1 = Dense(H * 6, activation='sigmoid')(reshape)
        dropout_1 = Dropout(0.5)(hidden1)
        hidden2 = Dense(H * 6, activation='sigmoid')(dropout_1)
        output = Dense(S, activation='sigmoid')(hidden2)

        return output
                
    #***************** | __model() | *******************#
    # El proposito de esta función es generar el modelo #
    # para resolver el problema CPMP con ayuda de capas #
    # de atención, una capa Flatten, una Dropout y      #
    # capaz Dense.                                      #
    #                                                   #
    # Input:                                            #
    #     - heads: número de cabezales que se usarán    #
    #              en la capa de atención.              #
    #     - S: Cantidad máxima de stacks que aceptará   #
    #          el modelo.                               #
    #     - H: Altura máxima de los stacks que          #
    #          aceptará el modelo.                      #
    #     - optimizer: Optimizador que usará el         #
    #                  modelo a la hora del             #
    #                  entrenamiento.                   #   
    # Output:                                           #
    #     Retorna el modelo capaz de resolver el        #
    #     problema CPMP.                                #

    def create_model(self, num_layer_attention_add: int = 1,
                heads: int = 5, S: int = 5, H: int = 5,
                optimizer: str | None = 'Adam'
                ) -> Model:
        input = Input(shape=(S, H + 1), dtype= tf.float32)

        model_so = Model_CPMP(num_layer_attention_add, heads, S, H)(input)
        expand = LayerExpandOutput()(model_so)
        concat = ConcatenationLayer()([input, model_so])
        model_sd = Model_CPMP(num_layer_attention_add, heads, S= S, H= H + 1)

        distributed = TimeDistributed(model_sd)(concat)

        flatten = Flatten()(distributed)

        output = OutputMultiplication()(flatten, expand)
    
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics= ['mae', 'mse'])

        self.__model = model

    def set_models(self, name: str) -> None:
        self.__model = load_model(name)

        input_shape = self.__model.layers[0].input_shape[0]
        self.__S = input_shape[1]
        self.__H = input_shape[2] - 1

    def get_dim(self) -> tuple:
        return self.__S, self.__H

    def save_models(self, name) -> None:
        if self.__model is None:
            print('Model have not been initialized.')
            return

        self.__model.save(name)

    def fit(self, X_train: np.ndarray, y_label: np.ndarray,
            batch_size: int = 32, epochs: int = 1, verbose: bool = True
            ) -> tuple:
        if self.__model is None:
            print('Model have not been initialized.')
            return
        
        historial = self.__model.fit(X_train, y_label, batch_size= batch_size, 
                                     epochs= epochs, verbose= verbose)
        
        return historial

    def predict(self, state: np.ndarray) -> tuple:
        if self.__model is None:
            print('Model have not been initialized.')
            return

        return self.__model.predict(state)

    def plot_model(self, name: str = 'model', show_shapes: bool = True) -> None:
        if self.__model is None:
            print('Model have not been initialized.')
            return

        name_img = name + str(self.__S) + 'x' + str(self.__H) + '.png'

        keras.utils.plot_model(self.__model, show_shapes=show_shapes,
                               to_file= name_img)

        imagen = plt.imread(name_img)
        plt.figure(figsize=(20, 25))
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()
