import sys
import os
import math
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def get_images(path):
    """
    This function returns the list of files present in the given path
    """
    list_images = os.listdir(path)

    return list_images # return the list of filename of the images


def preprocess_image(img, target_size):
    """
    This function will preprocess the image in the encoding of the image as per the given model
    """
    img = image.load_img(img, target_size=target_size) # load image using keras
    img = image.img_to_array(img) # convert the image into numpy array
    img = img.reshape((1,224,224,3)) # reshape the image according to the shape of the pretrained model

    return preprocess_input(img) # return the preprocess input image for our model


def show_image(image):
    plt.imshow(image.reshape((224,224,3)))


