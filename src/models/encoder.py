# src/models/encoder.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU

def build_encoder(input_dim, latent_dim, layers=[128, 64], dropout_rate=0.5):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for units in layers:
        x = Dense(units)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    encoded = Dense(latent_dim, activation='relu')(x)
    return tf.keras.Model(inputs=input_layer, outputs=encoded)
