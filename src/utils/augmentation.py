# src/utils/augmentation.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(data, augmentation_params=None):
    """Augment data if it's image data."""
    if augmentation_params is None:
        augmentation_params = {
            'rotation_range': 20,
            'zoom_range': 0.15,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.15,
            'horizontal_flip': True,
            'fill_mode': "nearest"
        }
    datagen = ImageDataGenerator(**augmentation_params)
    return datagen.flow(data)
