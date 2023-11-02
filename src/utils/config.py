# src/utils/config.py

import json
import os

def save_config(config, config_name, config_dir='configs'):
    """Save configuration as a JSON file."""
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    path = os.path.join(config_dir, f"{config_name}.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(config_name, config_dir='configs'):
    """Load configuration from a JSON file."""
    path = os.path.join(config_dir, f"{config_name}.json")
    with open(path, 'r') as f:
        return json.load(f)
