from utils.architectures import MLP, MLP2, CNN3D, VGG3D
from utils.models import ResNet3D

def mlp(output_layers, input_shape=(48,192,192)):
    NET = MLP(input_shape=input_shape,output_layers=output_layers)

    return NET.GetModel()

def mlp2(output_layers, input_shape=(48,192,192)):
    NET = MLP2(input_shape=input_shape,output_layers=output_layers)

    return NET.GetModel()

def cnn3d(output_layers, input_shape=(48,192,192)):
    NET = CNN3D(input_shape=input_shape,output_layers=output_layers)

    return NET.GetModel()

def resnet3d(output_layers, input_shape=(48,192,192)):
    NET = ResNet3D(input_shape=input_shape,output_layers=output_layers)

    return NET.GetModel()

def vgg3d(output_layers, input_shape=(48,192,192)):
    NET = VGG3D(input_shape=input_shape, output_layers=output_layers)

    return NET.GetModel()