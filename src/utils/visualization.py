# src/utils/visualization.py

import matplotlib.pyplot as plt

def plot_training_history(history, metrics=['loss', 'accuracy']):
    """Plot training and validation metrics."""
    fig, ax = plt.subplots(len(metrics), 1, figsize=(12, 8))
    for i, metric in enumerate(metrics):
        ax[i].plot(history.history[metric], label=f'Training {metric.capitalize()}')
        ax[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        ax[i].legend()
    plt.show()
