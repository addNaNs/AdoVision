import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


TRAINING_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/train"
VALIDATION_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/validation"

'''
digits_dirs = [os.path.join('../data/japanese_handwritten_digits/Japanese Handwritten Digits/train/0'
                            + str(i)) for i in range(10)]

[print('total ' + str(i) + ' images:', len(os.listdir(digits_dirs[i]))) for i in range(10)]
digits_files = [os.listdir(d) for d in digits_dirs]

TRAINING_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/train"
training_data_gen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')

VALIDATION_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/validation"
validation_data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_data_gen.flow_from_directory(
    TRAINING_DIR,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical'
)'''

import quick_tests.gym
gym = quick_tests.gym.Gym(TRAINING_DIR, VALIDATION_DIR)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

gym.set_model(model)
gym.get_model().compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
gym.fit_model()


