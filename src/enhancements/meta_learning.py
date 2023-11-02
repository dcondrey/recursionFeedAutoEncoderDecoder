# src/enhancements/meta_learning.py

from kerastuner.tuners import Hyperband

def hyperparameter_tuning(autoencoder, input_data, output_data):
    """
    Use Hyperband for hyperparameter tuning.
    autoencoder: the autoencoder model.
    input_data: training data.
    output_data: target data.
    """
    def model_builder(hp):
        lr = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        latent_dim = hp.Int('latent_dim', min_value=32, max_value=256, step=32)
        
        encoder = build_encoder(input_data.shape[1], latent_dim)
        decoder = build_decoder(latent_dim, input_data.shape[1])
        
        autoencoder = autoencoder_with_recursion(input_data.shape[1], latent_dim, 3)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')
        return autoencoder

    tuner = Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=10,
        factor=3,
        directory='hyperband_logs',
        project_name='autoencoder_tuning'
    )

    tuner.search(input_data, output_data, epochs=10, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model
