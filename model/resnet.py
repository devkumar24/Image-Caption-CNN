import sys
import os
import math

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import decode_predictions
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications import *
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.engine import input_layer
from keras.models import Sequential
from keras.models import Model

def resnet50(weights : str = "", input_shape = (224,224,3), summary = True):
    model = ResNet50(weights=weights, input_shape=input_shape)

    if summary:
        model.summary()

    return model

def vgg19(weights : str = "", input_shape = (224,224,3), summary = True):
    model = VGG19(weights=weights, input_shape=input_shape)

    if summary:
        model.summary()

    return model

def mobilenet(weights : str = "", input_shape = (224,224,3), summary = True):
    model = MobileNet(weights=weights, input_shape=input_shape)

    if summary:
        model.summary()

    return model

def inception(weights : str = "", input_shape = (224,224,3), summary = True):
    model = InceptionV3(weights=weights, input_shape=input_shape)

    if summary:
        model.summary()

    return model

def get_model(model, **kwargs):
    """
    input_layer : model.input
    output_layer : model.layers[-2].output

    returns a functional model that takes a custom input layer, custom output layer
    """
    keys = kwargs.keys()
    if len(keys) > 2 :
        if ("input_layer" in keys) and ("output_layer" in keys) :
            input_layer = kwargs['input_layer']
            output_layer = kwargs['output_layer']

            return Model(input_layer, output_layer)

    return Model(model.input, model.layers[-2].output)

def summary(model):
    model.summary()



