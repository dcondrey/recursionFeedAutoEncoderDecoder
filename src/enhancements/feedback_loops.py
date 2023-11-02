# src/enhancements/feedback_loops.py

import tensorflow as tf
from utils import save_model, load_model, setup_logger

logger = setup_logger("feedback_loops")

def compute_reconstruction_error(original_data, reconstructed_data):
    """
    Compute the reconstruction error between the original and reconstructed data.
    original_data: original input data.
    reconstructed_data: output from the autoencoder.
    """
    return tf.keras.losses.mean_squared_error(original_data, reconstructed_data)

def adjust_learning_rate(optimizer, error, previous_error, lr_increase_factor=1.1, lr_decrease_factor=0.9):
    """
    Adjust the learning rate based on the reconstruction error.
    optimizer: the optimizer used in training.
    error: current reconstruction error.
    previous_error: reconstruction error from the previous iteration.
    lr_increase_factor: factor by which to increase the learning rate.
    lr_decrease_factor: factor by which to decrease the learning rate.
    """
    current_lr = float(optimizer.lr.numpy())
    if error < previous_error:
        new_lr = current_lr * lr_increase_factor
    else:
        new_lr = current_lr * lr_decrease_factor
    optimizer.lr.assign(new_lr)
    logger.info(f"Adjusted learning rate from {current_lr} to {new_lr}")

def rollback_model_if_needed(model, error, threshold, checkpoint_dir='checkpoints'):
    """
    Rollback the model to a previous state if the error exceeds a threshold.
    model: the current model.
    error: current reconstruction error.
    threshold: error threshold for rollback.
    checkpoint_dir: directory where model checkpoints are saved.
    """
    if error > threshold:
        model = load_model("autoencoder_checkpoint", checkpoint_dir)
        logger.warning(f"Model rolled back due to error exceeding threshold. Current error: {error}")

def refine_model_based_on_error(model, error, low_threshold, high_threshold):
    """
    Refine the model architecture based on the reconstruction error.
    model: the current model.
    error: current reconstruction error.
    low_threshold: error threshold below which the model is considered to be performing well.
    high_threshold: error threshold above which the model is considered to be performing poorly.
    """
    if error < low_threshold:
        # Example: Remove a layer if the model is performing well
        model.pop()
        logger.info("Removed a layer from the model based on low reconstruction error.")
    elif error > high_threshold:
        # Example: Add a layer if the model is performing poorly
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        logger.info("Added a layer to the model based on high reconstruction error.")
