from kerastuner.tuners import Hyperband, RandomSearch
import tensorflow as tf
from utils import setup_logger

logger = setup_logger("meta_learning")

def self_adjusting_hyperparameter_tuning(autoencoder, input_data, output_data, tuner_type='hyperband'):
    """
    Use specified tuner for hyperparameter tuning and allow the AI to adjust its own hyperparameters.
    autoencoder: the autoencoder model.
    input_data: training data.
    output_data: target data.
    tuner_type: type of tuner to use ('hyperband' or 'random_search').
    """
    if not input_data.shape[0] or not output_data.shape[0]:
        logger.error("Input data or output data are empty.")
        return None

    def model_builder(hp):
        try:
            lr = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
            dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
            latent_dim_choices = hp.Choice('latent_dim', values=[32, 64, 128, 256])
            encoder_layers = [hp.Int(f'encoder_layer_{i}_units', min_value=32, max_value=256, step=32) for i in range(2)]
            decoder_layers = [hp.Int(f'decoder_layer_{i}_units', min_value=32, max_value=256, step=32) for i in range(2)]
            
            encoder = build_encoder(input_data.shape[1], latent_dim_choices, encoder_layers, dropout_rate)
            decoder = build_decoder(latent_dim_choices, input_data.shape[1], decoder_layers, dropout_rate)
            
            autoencoder = autoencoder_with_recursion(input_data.shape[1], latent_dim_choices, 3)
            autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')
            return autoencoder
        except Exception as e:
            logger.error(f"Error during model building: {e}")
            return None

    try:
        if tuner_type == 'hyperband':
            tuner = Hyperband(
                model_builder,
                objective='val_loss',
                max_epochs=10,
                factor=3,
                directory='hyperband_logs',
                project_name='autoencoder_tuning'
            )
        elif tuner_type == 'random_search':
            tuner = RandomSearch(
                model_builder,
                objective='val_loss',
                max_trials=100,
                directory='random_search_logs',
                project_name='autoencoder_tuning'
            )
        else:
            logger.error(f"Unsupported tuner type: {tuner_type}")
            return None

        tuner.search(input_data, output_data, epochs=10, validation_split=0.2)
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        return None