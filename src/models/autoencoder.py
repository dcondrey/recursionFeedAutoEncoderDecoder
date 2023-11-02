# src/models/autoencoder.py

from .encoder import build_encoder
from .decoder import build_decoder
import tensorflow as tf

def autoencoder_with_recursion(input_dim, latent_dim, num_iterations):
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)
    
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    
    for _ in range(num_iterations - 1):
        decoded = encoder(decoded)
        decoded = decoder(decoded)
    
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder
