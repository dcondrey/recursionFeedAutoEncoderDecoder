# src/utils/data_loader.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path, file_type='csv'):
    """
    Load data from a file.
    file_path: path to the file.
    file_type: type of the file ('csv', 'excel', 'hdf', etc.).
    """
    loaders = {
        'csv': pd.read_csv,
        'excel': pd.read_excel,
        'hdf': pd.read_hdf
    }
    if file_type not in loaders:
        raise ValueError(f"Unsupported file type: {file_type}")
    return loaders[file_type](file_path)

def preprocess_data(data, method='minmax'):
    """
    Preprocess data: normalize or standardize values.
    data: input data (e.g., a pandas DataFrame).
    method: preprocessing method ('minmax' or 'standard').
    """
    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler()
    }
    if method not in scalers:
        raise ValueError(f"Unsupported preprocessing method: {method}")
    return scalers[method].fit_transform(data)

def split_data(data, labels, test_size=0.2, stratify=None):
    """
    Split data into training, validation, and test sets.
    data: input data.
    labels: target labels.
    test_size: proportion of the dataset to include in the test split.
    stratify: if not None, data is split in a stratified fashion.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=stratify)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train if stratify else None)
    return X_train, X_val, X_test, y_train, y_val, y_test
