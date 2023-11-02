# src/utils/metrics.py

import tensorflow as tf

def compute_reconstruction_error(original_data, reconstructed_data):
    """
    Compute the reconstruction error between the original and reconstructed data.
    original_data: original input data.
    reconstructed_data: output from the autoencoder.
    """
    return tf.keras.losses.mean_squared_error(original_data, reconstructed_data)

def compute_accuracy(true_labels, predictions):
    """
    Compute the accuracy of predictions.
    true_labels: true labels.
    predictions: model predictions.
    """
    return tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true_labels, predictions))

def compute_precision(true_labels, predictions):
    """
    Compute the precision of predictions.
    true_labels: true labels.
    predictions: model predictions.
    """
    return tf.reduce_mean(tf.keras.metrics.precision(true_labels, predictions))

def compute_recall(true_labels, predictions):
    """
    Compute the recall of predictions.
    true_labels: true labels.
    predictions: model predictions.
    """
    return tf.reduce_mean(tf.keras.metrics.recall(true_labels, predictions))

def compute_f1_score(true_labels, predictions):
    """
    Compute the F1 score of predictions.
    true_labels: true labels.
    predictions: model predictions.
    """
    precision = compute_precision(true_labels, predictions)
    recall = compute_recall(true_labels, predictions)
    return 2 * (precision * recall) / (precision + recall)
