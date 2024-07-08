from keras.layers import Layer
from keras.layers import Dense
from tensorflow import TensorArray

class CPMP_Masking(Layer):
    def __init__(self, H: int) -> None:
        super(CPMP_Masking, self).__init__()

        self.__dense_1__ = Dense(H, activation= 'sigmoid')
        self.__dense_2__ = Dense(H * 3, activation= 'sigmoid')
        self.__dense_3__ = Dense(H * 2, activation= 'sigmoid')
        self.__dense_4__ = Dense(H, activation= 'sigmoid')
    
    def call(self, arr: TensorArray) -> TensorArray:
        dense_1 = self.__dense_1__(arr)
        dense_2 = self.__dense_2__(dense_1)
        dense_3 = self.__dense_3__(dense_2)
        dense_4 = self.__dense_4__(dense_3)

        return dense_4