# src/utils/model_utils.py

import tensorflow as tf

def save_model(model, path):
    """Save the model architecture and weights."""
    model.save(path)

def load_model(path):
    """Load a saved model from a specified path."""
    return tf.keras.models.load_model(path)
