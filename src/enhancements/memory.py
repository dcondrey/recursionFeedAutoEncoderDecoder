# src/enhancements/memory.py

import tensorflow as tf

def compute_fisher_information(model, data, labels, num_samples=1000):
    """
    Compute the Fisher information for each parameter in the model.
    model: trained model.
    data: dataset used to compute Fisher information.
    num_samples: number of samples to estimate Fisher information.
    """
    fisher_information = [tf.zeros_like(var) for var in model.trainable_variables]
    for _ in range(num_samples):
        idx = tf.random.uniform(shape=(), minval=0, maxval=data.shape[0], dtype=tf.int32)
        sample_data = tf.expand_dims(data[idx], axis=0)
        sample_label = tf.expand_dims(labels[idx], axis=0)
        with tf.GradientTape() as tape:
            logits = model(sample_data)
            loss = tf.keras.losses.categorical_crossentropy(sample_label, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        for i, grad in enumerate(grads):
            fisher_information[i] += tf.square(grad)
    fisher_information = [info / num_samples for info in fisher_information]
    return fisher_information