import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


class Gym:

    """
    Training class for evaluating models, basic but upgradeable

    Contains the directories of training and validation data, one model that i may or may not evaluate, and
    file names and history for the models it had trained
    """

    train_gen = None
    validation_gen = None
    TRAINING_DIR = ''
    VALIDATION_DIR = ''
    model = None
    last_trained_his = None
    past_models = []

    def __init__(self, train_dir, validation_dir, **kwargs):
        """
        Creates data generators, and does data augmentation
        :param train_dir: training direcotry
        :param validation_dir: validation directory
        :param kwargs: kwargs for the flow_from_directory method
        """

        # set directory names
        self.TRAINING_DIR = train_dir
        self.VALIDATION_DIR = validation_dir

        # create generators, augmentation is suited for digits
        training_data_gen = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=10,
                                               zoom_range=0.1,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1)
        validation_data_gen = ImageDataGenerator(rescale=1. / 255)  # not augmenting validation data

        # creating the directory iterators
        self.train_gen = training_data_gen.flow_from_directory(self.TRAINING_DIR, **kwargs)
        self.validation_gen = validation_data_gen.flow_from_directory(self.VALIDATION_DIR, **kwargs)

    def get_model(self):
        return self.model

    def set_model(self, new_model):
        # remove record of the last trained model
        if new_model is not self.model:
            self.last_trained_his = None
        self.model = new_model

    def fit_model(self):
        self.last_trained_his = self.model.fit(self.train_gen,
                                               epochs=50, validation_data=self.validation_gen,
                                               verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

    def get_last_history(self):
        return self.last_trained_his

    def save_model(self, f_name):
        # save the model both as a file and in a list
        self.model.save(f_name)
        self.past_models.append((f_name, self.last_trained_his))

    def get_past_models(self):
        return self.past_models

    def show_history(self):
        # shows recorded models, and all data associated
        for f_name, his in self.past_models:
            print('"{}" - {}'.format(f_name, his.history))


if __name__ == '__main__':
    TRAINING_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/train"
    VALIDATION_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/validation"

    gym = Gym(TRAINING_DIR, VALIDATION_DIR, target_size=(28, 28), color_mode='grayscale', class_mode='categorical')

    models = list()
    '''
    models.append(tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))

    models.append(tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))'''

    '''
    models.append(tf.keras.models.Sequential([
           tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                  input_shape=(28, 28, 1), padding='same'),
           tf.keras.layers.Dropout(0.1),
           tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
           tf.keras.layers.Dropout(0.1),
           tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
           tf.keras.layers.Dropout(0.1),
           tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
           tf.keras.layers.Dropout(0.1),
           tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
           tf.keras.layers.Dropout(0.1),
           tf.keras.layers.Flatten(),
           tf.keras.layers.Dense(256, activation='relu'),
           tf.keras.layers.Dense(10, activation='softmax')
        ]))
    
    models.append(tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))

    models.append(tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))
    '''
    models.append(tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))

    for i, model in enumerate(models):
        gym.set_model(model)
        gym.get_model().compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        gym.fit_model()
        gym.save_model('jhd_weights' + str(i) + '.h5')

    gym.show_history()
