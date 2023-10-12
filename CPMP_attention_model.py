import keras
import numpy as np
from keras.layers import Input, Flatten
from keras.layers import TimeDistributed
from keras.models import Model, load_model
from Layers import ConcatenationLayer, LayerExpandOutput, OutputMultiplication, Model_CPMP
import matplotlib.pyplot as plt
import tensorflow as tf

class CPMP_attention_model():
    def __init__(self) -> None:
        self.__S = 0
        self.__H = 0
        self.__model = None

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

    def set_model(self, name: str) -> None:
        self.__model = load_model(name, custom_objects= {'Model_CPMP': Model_CPMP, 
                                                         'OutputMultiplication': OutputMultiplication,
                                                         'LayerExpandOutput': LayerExpandOutput,
                                                         'ConcatenationLayer': ConcatenationLayer})

        input_shape = self.__model.layers[0].input_shape[0]
        self.__S = input_shape[1]
        self.__H = input_shape[2] - 1

    def get_dim(self) -> tuple:
        return self.__S, self.__H

    def save_model(self, name) -> None:
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
