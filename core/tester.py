import os
import struct

import keras_preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras_preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('jhd_weights.h5')

path = input()
img = keras_preprocessing.image.load_img(path, target_size=(28, 28), color_mode='grayscale')
plt.imshow(img)
plt.show()
x = keras_preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = 1 - x / 255.

images = np.vstack([x])
classes = model.predict(images)
print(path)
print(classes)
