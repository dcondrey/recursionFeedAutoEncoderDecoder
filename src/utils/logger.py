# src/utils/logging_util.py

import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up logging to file and console.
    track the model's progress, anomalies, and other significant events during training and evaluation.
    """
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
