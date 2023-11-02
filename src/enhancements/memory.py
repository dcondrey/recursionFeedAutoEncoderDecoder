# src/enhancements/memory.py

import tensorflow as tf

def compute_fisher_information(model, data, num_samples=1000):
    """
    Compute the Fisher information for each parameter in the model.
    model: trained model.
    data: dataset used to compute Fisher information.
    num_samples: number of samples to estimate Fisher information.
    """
    fisher_information = []
    for v in range(len(model.trainable_variables)):
        fisher_information.append(tf.zeros_like(model.trainable_variables[v]))

    for _ in range(num_samples):
        inputs = data.sample(1)
        with tf.GradientTape() as tape:
            outputs = model(inputs)
        gradients = tape.gradient(outputs, model.trainable_variables)
        for v in range(len(fisher_information)):
            fisher_information[v] += tf.square(gradients[v])

    fisher_information = [info / num_samples for info in fisher_information]
    return fisher_information
