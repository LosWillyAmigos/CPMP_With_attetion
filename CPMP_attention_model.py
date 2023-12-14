import keras
import numpy as np
from keras.layers import Input, Add, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from keras.models import Model, load_model


class CPMP_attention_model():
    def __init__(self) -> None:
        self.__S = 0
        self.__H = 0
        self.__model_so = None
        self.__model_sd = None

    def __normalization_layer(self, attention: None, input: None) -> LayerNormalization:
        layer = Add()([input, attention])
        layer = LayerNormalization(epsilon=1e-6)(layer)

        return layer

    def __feed_forward_layer(self, input: None, num_neurons: int) -> Dense:
        # capa de feed para que el modelo pueda aprender
        layer = Dense(num_neurons, activation='sigmoid')(input)
        layer = Dense(num_neurons)(layer)

        return layer

    def __attention_layer(self, heads: int, d_model: int, reshape: None) -> MultiHeadAttention:
        attention = MultiHeadAttention(num_heads=heads, key_dim=d_model)(reshape, reshape)
        return attention

    def __stack_attention(self, heads: int, d_model: int, reshape: None, input: None) -> Dense:
        # por si se debe modificar la dimensión
        attention = self.__attention_layer(heads, d_model, reshape)
        normalization = self.__normalization_layer(input, attention)
        feed = self.__feed_forward_layer(normalization, d_model)

        return feed

    def __model(self, num_layer_attention_add: int = 1,
                heads: int = 5, S: int = 5, H: int = 5,
                optimizer: str | None = 'Adam'
                ) -> Model:
        input = Input(shape=(S, H + 1))

        reshape = self.__stack_attention(heads, H + 1, input, input)
        for i in range(num_layer_attention_add):
            reshape = self.__stack_attention(heads, H + 1, reshape, input)

        reshape = Flatten()(reshape)
        hidden1 = Dense(H * 6, activation='sigmoid')(reshape)
        dropout_1 = Dropout(0.5)(hidden1)
        hidden2 = Dense(H * 6, activation='sigmoid')(dropout_1)
        output = Dense(S, activation='softmax')(hidden2)

        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['mae', 'mse', 'accuracy'])

        return model

    def __concatenate_state(self, state: np.ndarray,
                            label: np.ndarray
                            ) -> np.ndarray:
        new_data = []

        for i in range(state.shape[0]):
            stack = state[i].tolist()
            stack.append(label[i])
            new_data.append(stack)

        return np.stack(new_data)

    ## ******* OBTENER DATOS PARA EL SEGUNDO MODELO ******* ##
    ## Concatenar los labels con el estado inicial para el  ##
    ## segundo modelo.                                      ##
    ## input:                                               ##
    ##      - state : estado actual en forma matricial.     ##
    ##      - labels : lista de etiquetas en forma de array.##
    def __states_with_labels(self, state: np.ndarray,
                             labels: np.ndarray
                             ) -> np.ndarray:
        list_states = []

        for label in labels:
            list_states.append(self.__concatenate_state(state[0], label))

        return np.array(list_states)

    ## ******** Pasar de vector flotante a binario ******** ##
    ## Recibe un vector de elementos flotantes, este análiza##
    ## Mediante el promedio si se transforma a 1 o 0 el ele-##
    ## mento i-ésimo.                                       ##
    def __analyzer(self, arr: np.ndarray) -> np.ndarray:
        mean = np.mean(arr)
        labels = []
        for i in range(arr.shape[0]):
            if arr[i] < mean:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(labels, dtype='float64')

    def __destination_analyzer(self, labels: np.ndarray) -> np.ndarray:
        destination_list = []
        for _ in labels:
            print(_)
            destination_list.append(self.__analyzer(_))

        return np.array(destination_list)

    ## ******** OBTENER CATEGORIA DE MODELO ORIGEN ******** ##
    ## Toma el arreglo de predicciones y luego convierte a  ##
    ## etiquetas categoricas.                               ##
    ## input  :                                             ##
    ##  - arr : arreglo de valores entre 0 y 1.             ##
    ##  - sep : cambia el valor de la etiqueta según la se- ##
    ##        paración.                                     ##
    ## output : arreglo de arreglos con las etiquetas con-  ##
    ##        vertidas.                                     ##
    def __label_categorical(self, arr: np.ndarray = None) -> np.ndarray:
        list_categorical = []
        label = self.__analyzer(arr)
        idx_labels = np.where(label == 1)[0]

        for idx in idx_labels:
            list_categorical.append(self.__to_labels(idx))

        return np.array(list_categorical)

    def __to_labels(self, idx: np.ndarray) -> np.ndarray:
        arr = np.zeros(self.__S)
        arr[idx] = 1

        return arr

    def set_models(self, name_so: str, name_sd: str) -> None:
        self.__model_so = load_model(name_so)
        self.__model_sd = load_model(name_sd)

        input_shape = self.__model_so.layers[0].input_shape[0]
        self.__S = input_shape[1]
        self.__H = input_shape[2] - 1

    def get_dim(self) -> tuple:
        return self.__S, self.__H

    def save_models(self, name_so: str, name_sd: str) -> None:
        if self.__model_so is None or self.create_model_sd is None:
            print('Models have not been initialized.')
            return

        self.__model_so.save(name_so)
        self.__model_sd.save(name_sd)

    def save_model_so(self, name: str) -> None:
        if self.__model_so is None:
            print('Origin Model have not been initialized.')
            return

        self.__model_so.save(name)

    def save_model_sd(self, name: str) -> None:
        if self.__model_sd is None:
            print('Destination Model have not been initialized.')
            return

        self.__model_sd.save(name)

    def fit(self, X_train_so: np.ndarray, X_train_sd: np.ndarray,
            y_label_so: np.ndarray, y_label_sd: np.ndarray, batch_size: int = 32,
            epochs: int = 1, verbose: bool = True
            ) -> tuple:
        if self.__model_so is None or self.create_model_sd is None:
            print('Models have not been initialized.')
            return

        if verbose:
            print("Historial SO:")
        historial_so = self.__model_so.fit(X_train_so, y_label_so, batch_size=batch_size, epochs=epochs,
                                           verbose=verbose)
        if verbose:
            print("\nHistorial SD:")
        historial_sd = self.__model_sd.fit(X_train_sd, y_label_sd, batch_size=batch_size, epochs=epochs,
                                           verbose=verbose)

        return historial_so, historial_sd

    def fit_so(self, X_train_so: np.ndarray, y_label_so: np.ndarray,
               batch_size: int = 32, epochs: int = 1, verbose: bool = True
               ) -> list:
        if self.__model_so is None:
            print('Origin Model have not been initialized.')
            return

        historial_so = self.__model_so.fit(X_train_so, y_label_so, batch_size=batch_size, epochs=epochs,
                                           verbose=verbose)

        return historial_so

    def fit_sd(self, X_train_sd: np.ndarray, y_label_sd: np.ndarray,
               batch_size: int = 32, epochs: int = 1, verbose: bool = True
               ) -> list:
        if self.__model_sd is None:
            print('Destination Model have not been initialized.')
            return

        historial_sd = self.__model_sd.fit(X_train_sd, y_label_sd, batch_size=batch_size, epochs=epochs,
                                           verbose=verbose)

        return historial_sd

    def predict(self, state: np.ndarray) -> tuple:
        if self.__model_so is None or self.__model_sd is None:
            print('Models have not been initialized.')
            return

        origin_labels = self.__model_so.predict(state)
        origin_labels_categorical = self.__label_categorical(origin_labels)

        states = self.__states_with_labels(state, origin_labels_categorical)

        destination_labels = self.__model_sd(states)
        destination_labels_categorical = self.__destination_analyzer(destination_labels)

        return origin_labels_categorical, destination_labels_categorical

    def predict_so(self, states: np.ndarray) -> np.ndarray:
        return self.__model_so.predict(states)

    def predict_sd(self, states: np.ndarray) -> np.ndarray:
        return self.__model_sd.predict(states)

    def plot_model(self, name: str = 'model', show_shapes: bool = True) -> None:
        if self.__model_so is None or self.__model_sd is None:
            print('Models have not been initialized.')
            return

        keras.utils.plot_model(self.__model_so, show_shapes=show_shapes,
                               to_file=name + '_so_' + str(self.__S) + 'x' + str(self.__H) + '.png')
        keras.utils.plot_model(self.__model_sd, show_shapes=show_shapes,
                               to_file=name + '_sd_' + str(self.__S) + 'x' + str(self.__H) + '.png')

    def plot_model_so(self, name: str = 'model', show_shapes: bool = True) -> None:
        if self.__model_so is None:
            print('Model have not been initialized.')
            return

        keras.utils.plot_model(self.__model_so, show_shapes=show_shapes,
                               to_file=name + '_so_' + str(self.__S) + 'x' + str(self.__H) + '.png')

    def plot_model_sd(self, name: str = 'model', show_shapes: bool = True) -> None:
        if self.__model_sd is None:
            print('Model have not been initialized.')
            return

        keras.utils.plot_model(self.__model_sd, show_shapes=show_shapes,
                               to_file=name + '_sd_' + str(self.__S) + 'x' + str(self.__H) + '.png')


    def create_model_sd(self, num_layer_attention_add: int = 1,
                        heads: int = 5, S: int = 5, H: int = 5,
                        optimizer: str | None = 'Adam'
                        ) -> None:
        self.__S = S
        self.__H = H

        self.__model_sd = self.__model(num_layer_attention_add=num_layer_attention_add,
                                       heads=heads, S=S, H=H,
                                       optimizer=optimizer)

    def create_model_so(self, num_layer_attention_add: int = 1,
                        heads: int = 5, S: int = 5, H: int = 5,
                        optimizer: str | None = 'Adam'
                        ) -> None:
        self.__S = S
        self.__H = H

        self.__model_so = self.__model(num_layer_attention_add=num_layer_attention_add,
                                       heads=heads, S=S, H=H,
                                       optimizer=optimizer)

    def create_model(self, num_layer_attention_add_so: int = 1,
                     num_layer_attention_add_sd: int = 1,
                     heads_so: int = 5, heads_sd: int = 5, S: int = 5, H: int = 5,
                     optimizer: str | None = 'Adam'
                     ) -> None:
        self.__S = S
        self.__H = H

        self.__model_so = self.__model(num_layer_attention_add=num_layer_attention_add_so,
                                       heads=heads_so, S=S, H=H,
                                       optimizer=optimizer)
        self.__model_sd = self.__model(num_layer_attention_add=num_layer_attention_add_sd,
                                       heads=heads_sd, S=S, H=H + 1,
                                       optimizer=optimizer)
