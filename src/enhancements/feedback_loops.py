# src/enhancements/feedback_loops.py

import tensorflow as tf

def compute_reconstruction_error(original_data, reconstructed_data):
    """
    Compute the reconstruction error between the original and reconstructed data.
    original_data: original input data.
    reconstructed_data: output from the autoencoder.
    """
    return tf.keras.losses.mean_squared_error(original_data, reconstructed_data)
