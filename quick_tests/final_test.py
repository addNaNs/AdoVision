import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    TRAINING_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/train"
    VALIDATION_DIR = "../data/japanese_handwritten_digits/Japanese Handwritten Digits/validation"

    training_data_gen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=10,
                                           zoom_range=0.1,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1)
    validation_data_gen = ImageDataGenerator(rescale=1. / 255)  # not augmenting validation data

    # creating the directory iterators
    train_gen = training_data_gen.flow_from_directory(TRAINING_DIR, target_size=(28, 28),
                                                      color_mode='grayscale', class_mode='categorical')
    validation_gen = validation_data_gen.flow_from_directory(VALIDATION_DIR, target_size=(28, 28),
                                                             color_mode='grayscale', class_mode='categorical')

    model = tf.keras.models.Sequential([
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
        ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_gen,
              epochs=50, validation_data=validation_gen,
              verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    model.save('model.h5')
