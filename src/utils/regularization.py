# src/utils/regularization.py

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, BatchNormalization

def apply_regularization(model, l1_value=None, l2_value=None, dropout_rate=None, batch_norm=False):
    """
    Apply various regularization techniques to the model.
    
    Parameters:
    - model: The input model to which regularization will be applied.
    - l1_value: The regularization strength for L1 regularization. If None, L1 regularization is not applied.
    - l2_value: The regularization strength for L2 regularization. If None, L2 regularization is not applied.
    - dropout_rate: The dropout rate for Dropout layers. If None, Dropout is not applied.
    - batch_norm: A boolean indicating whether to apply Batch Normalization.
    
    Returns:
    - The model with the specified regularization techniques applied.
    """
    regularized_model = tf.keras.Sequential()
    
    for layer in model.layers:
        # Apply L1 and/or L2 regularization if specified
        if hasattr(layer, 'kernel_regularizer') and (l1_value or l2_value):
            reg = None
            if l1_value and l2_value:
                reg = regularizers.l1_l2(l1=l1_value, l2=l2_value)
            elif l1_value:
                reg = regularizers.l1(l1_value)
            elif l2_value:
                reg = regularizers.l2(l2_value)
            layer.kernel_regularizer = reg
        
        regularized_model.add(layer)
        
        # Add Dropout layer if specified
        if dropout_rate and not isinstance(layer, Dropout):
            regularized_model.add(Dropout(dropout_rate))
        
        # Add BatchNormalization layer if specified
        if batch_norm and not isinstance(layer, BatchNormalization):
            regularized_model.add(BatchNormalization())
    
    return regularized_model
