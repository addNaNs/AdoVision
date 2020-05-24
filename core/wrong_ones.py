import os
import struct

import keras_preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras_preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('jhd_weights.h5')

DIR = "../data/japanese_handwritten_digits/japanese handwritten digits/Japanese Handwritten Digits"
data_gen = keras_preprocessing.image.ImageDataGenerator(rescale=1. / 255)
data_gen = data_gen.flow_from_directory(DIR, target_size=(28, 28),
                                        color_mode='grayscale', class_mode='categorical')

pred = model.evaluate(data_gen)
print(pred)


