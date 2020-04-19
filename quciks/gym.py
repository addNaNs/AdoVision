import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


class Gym:

    train_gen = None
    validation_gen = None
    TRAINING_DIR = ''
    VALIDATION_DIR = ''
    model = None
    last_trained_his = None
    past_models = []

    def __init__(self, train_dir, validation_dir):
        self.TRAINING_DIR = train_dir
        self.VALIDATION_DIR = validation_dir

        training_data_gen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
        validation_data_gen = ImageDataGenerator(rescale=1. / 255)

        self.train_gen = training_data_gen.flow_from_directory(self.TRAINING_DIR,
                                                               target_size=(28, 28),
                                                               color_mode='grayscale',
                                                               class_mode='categorical')
        self.validation_gen = validation_data_gen.flow_from_directory(self.VALIDATION_DIR,
                                                                      target_size=(28, 28),
                                                                      color_mode='grayscale',
                                                                      class_mode='categorical')

    def get_model(self):
        return self.model

    def set_model(self, new_model):
        if new_model is not self.model:
            self.last_trained_his = None
        self.model = new_model

    def fit_model(self):
        self.last_trained_his = self.model.fit(self.train_gen,
                                               epochs=10, validation_data=self.validation_gen, verbose=1)

    def get_last_history(self):
        return self.last_trained_his

    def save_model(self, f_name):
        self.model.save(f_name)
        self.past_models.append((f_name, self.last_trained_his))


TRAINING_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/train"
VALIDATION_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/validation"

gym = Gym(TRAINING_DIR, VALIDATION_DIR)

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
gym.save_model('jhd_weights.h5')