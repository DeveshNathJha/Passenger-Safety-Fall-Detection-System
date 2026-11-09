"""
src/model_layers.py
-------------------
Custom Keras layers used inside the models. Kept separate from
signal_processing.py so the inference server never imports TensorFlow.

GravityAlignedFeatures turns raw phone-frame IMU windows into
orientation-invariant channels (see signal_processing.invariant_features
for the maths). Because it is a layer, it is exported into the .tflite file:
the server keeps sending raw sensor values and needs no extra code.
Only TFLite builtin ops are used (mean, mul, sum, sqrt, maximum, concat).
"""

import tensorflow as tf

_EPS = 1e-6


@tf.keras.utils.register_keras_serializable(package="fds")
class GravityAlignedFeatures(tf.keras.layers.Layer):
    """(batch, T, 6) raw acc/gyro → (batch, T, 6) rotation-invariant features."""

    def call(self, x):
        acc, gyr = x[..., 0:3], x[..., 3:6]
        g = tf.reduce_mean(acc, axis=1, keepdims=True)
        u = g / (tf.sqrt(tf.reduce_sum(g * g, axis=-1, keepdims=True)) + _EPS)

        def split(v):
            sq = tf.reduce_sum(v * v, axis=-1, keepdims=True)
            vert = tf.reduce_sum(v * u, axis=-1, keepdims=True)
            horiz = tf.sqrt(tf.maximum(sq - vert * vert, 0.0) + _EPS)
            return [tf.sqrt(sq + _EPS), vert, horiz]

        return tf.concat(split(acc) + split(gyr), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
