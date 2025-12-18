"""
Data Loading and Preprocessing Module
Handles dataset loading, missing value treatment, and preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(data_path='Data/agaricus-lepiota.data'):
    """
    Load the Mushroom dataset from local Data folder.
    
    Args:
        data_path: Path to the data file
    
    Returns:
        X: Features DataFrame
        y: Target Series
    """
    print("Loading Mushroom dataset from local Data folder...")
    
    # Column names based on agaricus-lepiota.names file
    column_names = [
        'class',
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color',
        'population', 'habitat'
    ]
    
    # Read the data file
    df = pd.read_csv(data_path, header=None, names=column_names, na_values='?')
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df[['class']].copy()
    y.columns = ['poisonous']
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y


def handle_missing_values(X):
    """
    Handle missing values by treating them as a separate category.
    Missing values are represented as '?' in the original data.
    """
    print("\nHandling missing values...")
    missing_counts = X.isna().sum()
    
    if missing_counts.sum() > 0:
        print(f"Missing values found in: {missing_counts[missing_counts > 0].to_dict()}")
        # Treat missing values as separate category
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna('missing')
        print("Missing values treated as separate category")
    else:
        print("No missing values found")
    
    return X


def split_data(X, y, test_size=0.30, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\nSplitting data into train/validation/test sets...")
    
    # First split: Train (70%) + Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: Validation (15%) + Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessing_pipeline(X_train, X_val, X_test):
    """
    Create preprocessing pipeline with OneHotEncoder and StandardScaler.
    
    Returns:
        pipeline: Fitted preprocessing pipeline
        X_train_dense: Preprocessed training data (dense array)
        X_val_dense: Preprocessed validation data (dense array)
        X_test_dense: Preprocessed test data (dense array)
    """
    print("\nCreating preprocessing pipeline...")
    
    categorical_features = X_train.columns.tolist()
    
    # ColumnTransformer for OneHotEncoding
    preprocess = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)]
    )
    
    # Full pipeline (OHE + StandardScaler)
    full_pipeline = Pipeline([
        ('preprocess', preprocess),
        ('scaler', StandardScaler(with_mean=False))
    ])
    
    # Fit on training data and transform all sets
    X_train_prep = full_pipeline.fit_transform(X_train)
    X_val_prep = full_pipeline.transform(X_val)
    X_test_prep = full_pipeline.transform(X_test)
    
    # Convert to dense arrays for models that require dense input
    X_train_dense = X_train_prep.toarray()
    X_val_dense = X_val_prep.toarray()
    X_test_dense = X_test_prep.toarray()
    
    print(f"Preprocessing complete. Feature count after encoding: {X_train_dense.shape[1]}")
    
    return full_pipeline, X_train_dense, X_val_dense, X_test_dense


def prepare_data():
    """
    Complete data preparation pipeline.
    
    Returns:
        Dictionary containing all prepared data splits and preprocessing pipeline
    """
    X, y = load_data()
    X = handle_missing_values(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline, X_train_dense, X_val_dense, X_test_dense = create_preprocessing_pipeline(
        X_train, X_val, X_test
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_train_dense': X_train_dense,
        'X_val_dense': X_val_dense,
        'X_test_dense': X_test_dense,
        'pipeline': pipeline
    }

