"""
Module for preprocessing UNSW_NB15 data for anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder #Encoding categorical features, normalizing numerical features.
from sklearn.model_selection import train_test_split #Splitting data into train/test sets.
import pickle #Utility for saving Python objects to files.
from pathlib import Path

def handle_missing_values(df, strategy = 'drop', fill_value = None):
    """Handle missing values in the DataFrame."""
    df_clean = df.copy() #copy is used to avoid modifying the original DataFrame, allowing for flexibility in experimentation with different strategies without altering the base data.
    if strategy == 'drop': #Drops rows with missing values. Can lead to data loss, yet simple. 
        df_clean = df_clean.dropna()
        print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
    elif strategy == 'mean': #Fills missing values with the mean of each column. Only suitable for numerical features, also can distort distributions.
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].mean())
        print("Filled missing values with mean")
    elif strategy == 'median': #Fills missing values with the median of each column. More robust to extreme values than mean imputation, but still can distort distributions.
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())
        print("Filled missing values with median")
    elif strategy == 'mode': #Fills missing values with the mode (most frequent value) of each column. Suitable for categorical features, but can lead to over-representation of the most common category.
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace = True) #We use inplace=True to modify the DataFrame directly.
    elif strategy == 'constant': #Fills missing values with a constant value.
        df_clean = df_clean.fillna(fill_value)
        print(f"Filled missing values with constant: {fill_value}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Strategy must be 'drop', 'mean', 'median', 'mode', or 'constant'.")
    return df_clean

def encode_categorical_features(df, categorical_cols, method = 'label', encoder = None):
    """Encode categorical features using specified method."""
    df_encoded = df.copy()
    encoder_dict = {} #Dict will store fitted encoders for each column.
    if method == 'label':
        for col in categorical_cols:
            if encoder and col in encoder: 
                # Use existing encoder (for test data)
                le = encoder[col]
            else:
                # Create new encoder (for train data)
                le = LabelEncoder()
                le.fit(df_encoded[col].astype(str))
            df_encoded[col] = le.transform(df_encoded[col].astype(str))
            encoder_dict[col] = le
        print(f"Label encoded {len(categorical_cols)} categorical features")        
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns = categorical_cols, prefix=categorical_cols)
        print(f"OH encoded {len(categorical_cols)} categorical features")
        print(f"New shape : {df_encoded.shape}")
    else:
        raise ValueError(f"Unknown encoding method : {method}")
    return df_encoded, encoder_dict  

def normalize_features(df, feature_cols, scaler = None):
    """Normalize numerical features using StandardScaler to get Mean = 0 and Std = 1."""
    df_normalized = df.copy()
    if scaler is None : 
        scaler = StandardScaler()
        df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
        print(f"Successfully fitted and transformed {len(feature_cols)} features")
    else : 
        df_normalized[feature_cols] = scaler.transform(df[feature_cols])
        print(f"Successfully transformed {len(feature_cols)} features using provided scaler")
    return df_normalized, scaler

def split_for_anomaly_detection(df, label_col = 'label', normal_label = 0, val_size = 0.2, test_size = 0.2, random_state = 42):
    """Split data into train/validation/test sets for anomaly detection.
    Train -> Only normal data.
    Validation -> ONLY NORMAL data (for monitoring overfitting).
    Test -> Mix of normal and anomalous data (for final evaluation)."""
    
    X = df.drop(columns = [label_col])
    y = df[label_col]
    
    # Separate normal and anomaly data
    normal_mask = (y == normal_label)
    X_normal = X[normal_mask]
    y_normal = y[normal_mask]
    X_anomaly = X[~normal_mask]
    y_anomaly = y[~normal_mask]
    
    # Split normal data into Train / Val / Test
    train_size = 1.0 - val_size - test_size
    
    # First split : Train vs (Val + Test)
    X_train, X_normal_remaining, y_train, y_normal_remaining = train_test_split(
        X_normal, y_normal, 
        test_size = (val_size + test_size),
        random_state = random_state
    )
    
    # Second split : Val vs Test (from remaining normal data)
    val_ratio = val_size / (val_size + test_size)
    
    X_val, X_test_normal, y_val, y_test_normal = train_test_split(
        X_normal_remaining, y_normal_remaining,
        test_size = (1 - val_ratio),
        random_state = random_state
    )
    
    # Test = Normal + All attacks
    if len(X_anomaly) > 0 :
        X_test = pd.concat([X_test_normal, X_anomaly], axis = 0)
        y_test = pd.concat([y_test_normal, y_anomaly], axis = 0)
        
        # Shuffle test set
        test_indices = np.random.RandomState(random_state).permutation(len(X_test))
        X_test = X_test.iloc[test_indices].reset_index(drop = True)
        y_test = y_test.iloc[test_indices].reset_index(drop = True)
    else :
        X_test = X_test_normal
        y_test = y_test_normal
    
    print("Data split summary :")
    print(f"Train set : {len(X_train) :6d} samples (100% normal)")
    print(f"Validation set : {len(X_val) :6d} samples (100% normal)")
    print(f"Test set : {len(X_test) :6d} samples ({(y_test == normal_label).sum()} normal, {(y_test != normal_label).sum()} attacks)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoders, output_dir = '../data/processed'):
    """Save preprocessed data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents = True, exist_ok = True)
    #Save data as numpy arrays 
    np.save(output_path / 'X_train.npy', X_train.values)
    np.save(output_path / 'X_val.npy', X_val.values)
    np.save(output_path / 'X_test.npy', X_test.values)
    np.save(output_path / 'y_train.npy', y_train.values)
    np.save(output_path / 'y_val.npy', y_val.values)
    np.save(output_path / 'y_test.npy', y_test.values)
    #Save scaler
    with open(output_path / 'scaler.pkl', 'wb') as f: 
        pickle.dump(scaler, f)
    #Save encoders
    with open(output_path / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    #Save feature names
    feature_names = X_train.columns.tolist()
    with open(output_path / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"- X_train.npy : {X_train.shape}, X_val.npy : {X_val.shape}, X_test.npy : {X_test.shape}")
    print(f"- y_train.npy : {y_train.shape}, y_val.npy : {y_val.shape}, y_test.npy : {y_test.shape}")
    print(f"- scaler.pkl, encoders.pkl, feature_names.pkl")
    
def load_preprocessed_data(input_dir = '../data/processed'):
    """Load preprocessed data."""
    input_path = Path(input_dir)
    data = {
        'X_train' : np.load(input_path / 'X_train.npy'),
        'X_val' : np.load(input_path / 'X_val.npy'),
        'X_test' : np.load(input_path / 'X_test.npy'),
        'y_train' : np.load(input_path / 'y_train.npy'),
        'y_val' : np.load(input_path / 'y_val.npy'),
        'y_test' : np.load(input_path / 'y_test.npy'),
    }
    with open(input_path / 'scaler.pkl', 'rb') as f:
        data['scaler'] = pickle.load(f)
    with open(input_path / 'encoders.pkl', 'rb') as f:
        data['encoders'] = pickle.load(f)
    with open(input_path / 'feature_names.pkl', 'rb') as f:
        data['feature_names'] = pickle.load(f)
    return data