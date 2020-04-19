import os
import pathlib
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

    def __init__(self, train_dir, validation_dir, **kwargs):
        self.TRAINING_DIR = train_dir
        self.VALIDATION_DIR = validation_dir

        training_data_gen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
        validation_data_gen = ImageDataGenerator(rescale=1. / 255)

        self.train_gen = training_data_gen.flow_from_directory(self.TRAINING_DIR, kwargs)
        self.validation_gen = validation_data_gen.flow_from_directory(self.VALIDATION_DIR, kwargs)

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
        self.history.append((f_name, self.last_trained_his))


