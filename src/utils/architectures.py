import tensorflow as tf
from tensorflow import keras

class mlp_1000():
    def __init__(
        self,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape

        self.model_input = keras.Input(shape=self.input_shape, name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Flatten()(self.model_input)
        x = keras.layers.Dense(1000, activation='relu')(x)

        out_seq   = keras.layers.Dense(5,activation="softmax",name="out_seq")(x)
        out_view  = keras.layers.Dense(3,activation="softmax",name="out_view")(x)
        out_ctrs  = keras.layers.Dense(2,activation="softmax",name="out_ctrs")(x)
        out_body  = keras.layers.Dense(9,activation="sigmoid",name="out_body")(x)

        return [out_seq, out_view, out_ctrs, out_body]

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

class cnn3d():
    def __init__(
        self,
        input_shape=(48,192,192,1),
        num_channels=1
        ):

        self.input_shape = input_shape
        self.num_channels = 1

        self.model_input = keras.Input(shape=self.input_shape, name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Conv3D(filters=64, kernel_size=(3,7,7), strides=(1,2,2), padding="same", name="conv1", activation=None)(self.model_input)
        x = keras.layers.BatchNormalization(name="bn1")(x)
        x = keras.layers.ReLU(name="relu1")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool1")(x)

        x = keras.layers.Conv3D(filters=128, kernel_size=(3,5,5), strides=(1,2,2), padding="same", name="conv2", activation=None)(x)
        # x = keras.layers.BatchNormalization(name="bn2")(x)
        x = keras.layers.ReLU(name="relu2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool2")(x)

        x = keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv3", activation=None)(x)
        # x = keras.layers.BatchNormalization(name="bn3")(x)
        x = keras.layers.ReLU(name="relu3")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool3")(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1000, activation="relu", name="last")(x)

        out_seq   = keras.layers.Dense(5,activation="softmax",name="out_seq")(x)
        out_view  = keras.layers.Dense(3,activation="softmax",name="out_view")(x)
        out_ctrs  = keras.layers.Dense(2,activation="softmax",name="out_ctrs")(x)
        out_body  = keras.layers.Dense(9,activation="sigmoid",name="out_body")(x)

        return [out_seq, out_view, out_ctrs, out_body]

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)