# src/models/encoder.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout

def build_encoder(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(latent_dim, activation='relu')(x)
    return tf.keras.Model(inputs=input_layer, outputs=encoded)
