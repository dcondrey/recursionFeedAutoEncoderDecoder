# src/models/decoder.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout

def build_decoder(latent_dim, output_dim):
    input_layer = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(output_dim, activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_layer, outputs=decoded)
