import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('jhd_weights0.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
open("jhd_lite0.tflite", "wb").write(quantized_model)


