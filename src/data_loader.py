"""
Utilities to load and inspect the UNSW-NB15 dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class UNSWNB15Loader:
    """Load train/test parquet files from a UNSW-NB15 data directory."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load_data(self):
        """Return train and test DataFrames."""
        train_df = load_unsw_data(self.data_dir, dataset='training')
        test_df = load_unsw_data(self.data_dir, dataset='testing')
        return train_df, test_df


def load_unsw_data(data_dir, dataset='training'):
    """Load one split of the UNSW-NB15 parquet dataset."""
    data_path = Path(data_dir)
    if dataset == 'training':
        file_path = data_path / 'UNSW_NB15_training-set.parquet'
    elif dataset == 'testing':
        file_path = data_path / 'UNSW_NB15_testing-set.parquet'
    else:
        raise ValueError("Dataset must be either 'training' or 'testing'.")

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def get_feature_types(df):
    """Return numerical and categorical column names."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {
        'numerical': numerical_features,
        'categorical': categorical_features,
    }


def get_data_summary(df):
    """Return a compact summary of the DataFrame."""
    return {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 ** 2),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
    }


__all__ = [
    'UNSWNB15Loader',
    'load_unsw_data',
    'get_feature_types',
    'get_data_summary',
]
