# src/enhancements/memory.py

import tensorflow as tf

def elastic_weight_consolidation_loss(old_task_weights, importance, model):
    """
    Compute the Elastic Weight Consolidation (EWC) loss.
    old_task_weights: weights from the previously learned task.
    importance: importance of the weights (Fisher information).
    model: current model.
    """
    ewc_loss = 0
    for idx, weight in enumerate(model.trainable_variables):
        ewc_loss += tf.reduce_sum(importance[idx] * tf.square(weight - old_task_weights[idx]))
    return ewc_loss
