"""
preprocessing.py
Data preprocessing module for the House Prices dataset.

This module handles:
- Loading the dataset
- Imputing missing values (numerical: median, categorical: "Missing")
- Encoding categorical variables using OrdinalEncoder
- Standardizing features only (NOT the target variable)
- Train-test split (80-20)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def load_data(filepath, random_state=42):
    """
    Load the House Prices dataset.
    
    Args:
        filepath (str): Path to train.csv
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
               - X_train: Features for training (standardized)
               - X_test: Features for testing (standardized)
               - y_train: Target variable for training (NOT standardized)
               - y_test: Target variable for testing (NOT standardized)
    """
    
    # Load data
    print("[INFO] Loading dataset from:", filepath)
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # Separate features and target
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    
    print(f"\n[INFO] Target variable (SalePrice):")
    print(f"  - Mean: {y.mean():.2f}")
    print(f"  - Std: {y.std():.2f}")
    print(f"  - Min: {y.min():.2f}")
    print(f"  - Max: {y.max():.2f}")
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n[INFO] Feature types:")
    print(f"  - Numerical features: {len(numerical_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    
    # Handle missing values: numerical columns → median imputation
    if len(numerical_cols) > 0:
        print(f"\n[INFO] Imputing numerical features with median...")
        numerical_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])
    
    # Handle missing values: categorical columns → fill with "Missing"
    if len(categorical_cols) > 0:
        print(f"[INFO] Imputing categorical features with 'Missing'...")
        for col in categorical_cols:
            X[col] = X[col].fillna("Missing")
    
    # Encode categorical variables using OrdinalEncoder
    if len(categorical_cols) > 0:
        print(f"[INFO] Encoding categorical features with OrdinalEncoder...")
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                         unknown_value=-1)
        X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
    
    # Verify no missing values remain
    missing_after = X.isnull().sum().sum()
    print(f"[INFO] Missing values after preprocessing: {missing_after}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"\n[INFO] Train-test split (80-20):")
    print(f"  - Training set size: {X_train.shape[0]}")
    print(f"  - Testing set size: {X_test.shape[0]}")
    print(f"  - Total features: {X_train.shape[1]}")
    
    # Standardize features only (NOT target variable)
    print(f"\n[INFO] Standardizing feature variables (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"[INFO] Standardization complete. Feature statistics:")
    print(f"  - Train X mean: {X_train.mean():.6f}")
    print(f"  - Train X std: {X_train.std():.6f}")
    
    return X_train, X_test, y_train, y_test, X.shape[1]