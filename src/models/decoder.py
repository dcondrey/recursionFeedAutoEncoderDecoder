# src/models/decoder.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU

def build_decoder(latent_dim, output_dim, layers=[64, 128], dropout_rate=0.5):
    input_layer = Input(shape=(latent_dim,))
    x = input_layer
    for units in layers:
        x = Dense(units)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    decoded = Dense(output_dim, activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_layer, outputs=decoded)
