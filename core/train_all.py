import os
import tensorflow as tf
import keras_preprocessing.image

if __name__ == '__main__':

    TRAINING_DIR = "../data/japanese_handwritten_digits/japanese handwritten digits/Japanese Handwritten Digits"
    training_data_gen = keras_preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                     rotation_range=10,
                                                                     zoom_range=0.1,
                                                                     width_shift_range=0.1,
                                                                     height_shift_range=0.1)
    train_gen = training_data_gen.flow_from_directory(TRAINING_DIR, target_size=(28, 28),
                                                      color_mode='grayscale', class_mode='categorical')

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_gen, epochs=20, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    model.save('jhd_weights.h5')
