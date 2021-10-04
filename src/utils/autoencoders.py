import tensorflow as tf
from tensorflow import keras
from networks import ResNet3D

class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", shape=[self.dense.input_shape[-1]], initializer="zeros")
        super().build(batch_input_shape)

    def call(self, input):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

class e1():
    def __init__(
        self,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape
        self.channels = 1

        self.model_input = keras.Input(shape=[*self.input_shape,self.channels], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        x = keras.layers.Conv3D(filters=32, kernel_size=(3,7,7), strides=(1,2,2), padding="same", name="conv1", activation=None)(self.model_input)
        # x = keras.layers.BatchNormalization(name="bn1")(x)
        x = keras.layers.ReLU(name="relu1")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool1")(x)
        # x = keras.layers.Conv3D(filters=64, kernel_size=(3,5,5), strides=(1,2,2), padding="same", name="conv2", activation=None)(x)
        # # x = keras.layers.BatchNormalization(name="bn1")(x)
        # x = keras.layers.ReLU(name="relu2")(x)
        # x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool2")(x)

        # x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(2048, activation="relu", name="last")(x)

        return x

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

class d1():
    def __init__(
        self,
        input_shape=(24,48,48,32)
        ):

        self.input_shape = input_shape
        self.model_input = keras.Input(shape=[*self.input_shape], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        # x = keras.layers.Conv3DTranspose(filters=32, kernel_size=(3,5,5), strides=(1,2,2), padding="same", name="conv2", activation=None)(self.model_input)
        # x = keras.layers.ReLU(name='relu2')(x)
        # x = keras.layers.UpSampling3D(size=(2,2,2))(x)
        x = keras.layers.Conv3DTranspose(filters=32, kernel_size=(3,7,7), strides=(1,2,2), padding="same", name="conv1", activation=None)(self.model_input)
        x = keras.layers.ReLU(name='relu1')(x)
        x = keras.layers.UpSampling3D(size=(2,2,2))(x)
        x = keras.layers.Conv3DTranspose(filters=1, kernel_size=(3,7,7), padding="same", name="output_layer", activation="sigmoid")(x)

        return x

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)

class lean_classifier():
    def __init__(
        self,
        output_layers,
        input_shape=(48,192,192)
        ):

        self.input_shape = input_shape
        self.output_layers = output_layers

        self.model_input = keras.Input(shape=[*self.input_shape], name="input_layer")
        self.model_output = self.BuildModel()

    def BuildModel(self):
        base_model = keras.models.load_model("./models/encoder")
        print(base_model.output.shape[1:])
        for layer in base_model.layers:
            layer.trainable = True
        x = base_model(self.model_input)
        x = keras.layers.Conv3D(filters=64, kernel_size=(3,5,5), strides=(1,2,2), padding="same", name="conv2", activation=None)(x)
        # x = keras.layers.BatchNormalization(name="bn2")(x)
        x = keras.layers.ReLU(name="relu2")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool2")(x)

        x = keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding="same", name="conv3", activation=None)(x)
        # x = keras.layers.BatchNormalization(name="bn3")(x)
        x = keras.layers.ReLU(name="relu3")(x)
        x = keras.layers.MaxPool3D(pool_size=(2,2,2), padding="valid", name="pool3")(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(2048, activation="relu", name="last")(x)

        out_list = []
        for idx in range(len(self.output_layers)):
            x_out = self.output_layers[idx](x)
            out_list.append(x_out)

        return out_list

    def GetModel(self):
        return tf.keras.Model(inputs=self.model_input,outputs=self.model_output)