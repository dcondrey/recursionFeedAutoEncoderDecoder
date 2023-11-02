# src/enhancements/exploration.py

import tensorflow as tf
import numpy as np
from utils import compute_reconstruction_error, setup_logger, save_model

logger = setup_logger("exploration")

def recursive_autoencoder_until_threshold(autoencoder, input_data, threshold=0.05, max_iterations=100, early_stopping_rounds=5, checkpoint_interval=10, save_dir='checkpoints'):
    """
    Enhanced recursive process with additional features.
    """
    iteration = 0
    consecutive_error_increase = 0
    previous_error = float('inf')
    error_rates = []

    reconstructed_data = autoencoder.predict(input_data)
    error = compute_reconstruction_error(input_data, reconstructed_data)

    while tf.reduce_mean(error) > threshold and iteration < max_iterations:
        if tf.reduce_mean(error) > previous_error:
            consecutive_error_increase += 1
        else:
            consecutive_error_increase = 0

        if consecutive_error_increase >= early_stopping_rounds:
            logger.warning(f"Early stopping at iteration {iteration} due to consecutive increase in error.")
            break

        # Checkpointing
        if iteration % checkpoint_interval == 0:
            save_model(autoencoder, f"autoencoder_checkpoint_{iteration}", save_dir)

        # Error rate logging and adaptive thresholding
        error_rates.append(tf.reduce_mean(error).numpy())
        if len(error_rates) > 2 and np.abs(error_rates[-1] - error_rates[-2]) < 0.001:  # Small change in error
            threshold *= 0.9  # Reduce threshold by 10%

        logger.info(f"Iteration {iteration}: Reconstruction Error = {tf.reduce_mean(error).numpy()}")
        
        reconstructed_data = autoencoder.predict(reconstructed_data)
        previous_error = tf.reduce_mean(error)
        error = compute_reconstruction_error(input_data, reconstructed_data)
        iteration += 1

    # Visualization
    plot_training_history({'reconstruction_error': error_rates})

    return reconstructed_data, iteration, tf.reduce_mean(error).numpy()
