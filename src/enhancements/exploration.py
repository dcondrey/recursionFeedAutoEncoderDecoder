# src/enhancements/exploration.py

def recursive_autoencoder_until_threshold(autoencoder, input_data, threshold=0.05, max_iterations=100):
    """
    Continue the recursive process until a certain reconstruction error threshold is met.
    autoencoder: the autoencoder model.
    input_data: input data for the autoencoder.
    threshold: reconstruction error threshold.
    max_iterations: maximum number of recursive iterations to prevent infinite loops.
    """
    iteration = 0
    reconstructed_data = autoencoder.predict(input_data)
    error = compute_reconstruction_error(input_data, reconstructed_data)

    while tf.reduce_mean(error) > threshold and iteration < max_iterations:
        reconstructed_data = autoencoder.predict(reconstructed_data)
        error = compute_reconstruction_error(input_data, reconstructed_data)
        iteration += 1

    return reconstructed_data
