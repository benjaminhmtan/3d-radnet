import tensorflow as tf
from tensorflow import keras
from networks import ResNet3D

class MLP():
    def __init__(
        self,
        output_layers,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape
        self.output_layers = output_layers

        self.model_input = keras.Input(shape=self.input_shape, name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Flatten()(self.model_input)
        x = keras.layers.Dense(750, activation='relu')(x)

        out_list = []
        for idx in range(len(self.output_layers)):
            x_out = self.output_layers[idx](x)
            out_list.append(x_out)
            # out_seq   = keras.layers.Dense(5,activation="softmax",name="out_seq")(x)
            # out_view  = keras.layers.Dense(3,activation="softmax",name="out_view")(x)
            # out_ctrs  = keras.layers.Dense(2,activation="softmax",name="out_ctrs")(x)
            # out_body  = keras.layers.Dense(9,activation="sigmoid",name="out_body")(x)

        return out_list

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

class MLP2():
    def __init__(
        self,
        output_layers,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape
        self.output_layers = output_layers

        self.model_input = keras.Input(shape=self.input_shape, name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Flatten()(self.model_input)
        x = keras.layers.Dense(750, activation='relu')(x)
        x = keras.layers.Dense(250, activation='relu')(x)

        out_list = []
        for idx in range(len(self.output_layers)):
            x_out = self.output_layers[idx](x)
            out_list.append(x_out)
            # out_seq   = keras.layers.Dense(5,activation="softmax",name="out_seq")(x)
            # out_view  = keras.layers.Dense(3,activation="softmax",name="out_view")(x)
            # out_ctrs  = keras.layers.Dense(2,activation="softmax",name="out_ctrs")(x)
            # out_body  = keras.layers.Dense(9,activation="sigmoid",name="out_body")(x)

        return out_list

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

class CNN3D():
    def __init__(
        self,
        output_layers,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape
        self.output_layers = output_layers
        self.channels = 1

        self.model_input = keras.Input(shape=self.input_shape+(self.channels,), name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Conv3D(filters=32, kernel_size=(3,7,7), strides=(1,2,2), padding="same", name="conv1", activation=None)(self.model_input)
        x = keras.layers.BatchNormalization(name="bn1")(x)
        x = keras.layers.ReLU(name="relu1")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool1")(x)

        x = keras.layers.Conv3D(filters=64, kernel_size=(3,5,5), strides=(1,2,2), padding="same", name="conv2", activation=None)(x)
        # x = keras.layers.BatchNormalization(name="bn2")(x)
        x = keras.layers.ReLU(name="relu2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool2")(x)

        x = keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv3", activation=None)(x)
        # x = keras.layers.BatchNormalization(name="bn3")(x)
        x = keras.layers.ReLU(name="relu3")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool3")(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1000, activation="relu", name="last")(x)

        out_list = []
        for idx in range(len(self.output_layers)):
            x_out = self.output_layers[idx](x)
            out_list.append(x_out)

        # out_seq   = keras.layers.Dense(5,activation="softmax",name="out_seq")(x)
        # out_view  = keras.layers.Dense(3,activation="softmax",name="out_view")(x)
        # out_ctrs  = keras.layers.Dense(2,activation="softmax",name="out_ctrs")(x)
        # out_body  = keras.layers.Dense(9,activation="sigmoid",name="out_body")(x)

        return out_list

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

class VGG3D():
    def __init__(
        self,
        output_layers,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape
        self.output_layers = output_layers
        self.channels = 1

        self.model_input = keras.Input(shape=self.input_shape+(self.channels,), name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv1-1", activation=None)(self.model_input)
        x = keras.layers.BatchNormalization(name="bn1-1")(x)
        x = keras.layers.ReLU(name="relu1-1")(x)
        x = keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv1-2", activation=None)(self.model_input)
        x = keras.layers.BatchNormalization(name="bn1-2")(x)
        x = keras.layers.ReLU(name="relu1-2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool1")(x)

        x = keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv2-1", activation=None)(x)
        x = keras.layers.BatchNormalization(name="bn2-1")(x)
        x = keras.layers.ReLU(name="relu2-1")(x)
        x = keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv2-2", activation=None)(x)
        x = keras.layers.BatchNormalization(name="bn2-2")(x)
        x = keras.layers.ReLU(name="relu2-2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool2")(x)

        x = keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv3-1", activation=None)(x)
        x = keras.layers.BatchNormalization(name="bn3-1")(x)
        x = keras.layers.ReLU(name="relu3-1")(x)
        x = keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv3-2", activation=None)(x)
        x = keras.layers.BatchNormalization(name="bn3-2")(x)
        x = keras.layers.ReLU(name="relu3-2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool3")(x)

        x = keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv4-1", activation=None)(x)
        x = keras.layers.BatchNormalization(name="bn4-1")(x)
        x = keras.layers.ReLU(name="relu4-1")(x)
        x = keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv4-2", activation=None)(x)
        x = keras.layers.BatchNormalization(name="bn4-2")(x)
        x = keras.layers.ReLU(name="relu4-2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool4")(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096, activation="relu", name="last")(x)

        out_list = []
        for idx in range(len(self.output_layers)):
            x_out = self.output_layers[idx](x)
            out_list.append(x_out)

        # out_seq   = keras.layers.Dense(5,activation="softmax",name="out_seq")(x)
        # out_view  = keras.layers.Dense(3,activation="softmax",name="out_view")(x)
        # out_ctrs  = keras.layers.Dense(2,activation="softmax",name="out_ctrs")(x)
        # out_body  = keras.layers.Dense(9,activation="sigmoid",name="out_body")(x)

        return out_list

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)