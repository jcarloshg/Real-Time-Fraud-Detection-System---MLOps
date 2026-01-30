"""
Data preparation script for fraud detection model training.
Loads raw data, splits it, scales features, and saves processed data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path

def load_and_prepare_data(
    data_path='../../data/raw/creditcard.csv',
    test_size=0.2,
    validation_size=0.1,
    random_state=42
):
    """
    Load and prepare the fraud detection dataset
    
    Args:
        data_path: Path to the raw CSV file
        test_size: Proportion of data for test set
        validation_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    
    print("="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   ✅ Dataset loaded: {df.shape}")
    
    # Basic statistics
    fraud_count = df['Class'].sum()
    fraud_rate = df['Class'].mean() * 100
    print(f"   📊 Total transactions: {len(df):,}")
    print(f"   📊 Fraud cases: {fraud_count:,} ({fraud_rate:.2f}%)")
    
    # Separate features and target
    print("\n2. Separating features and target...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   ✅ Target shape: {y.shape}")
    
    # First split: separate test set
    print(f"\n3. Splitting data (test_size={test_size})...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Second split: separate validation set from training set
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"   ✅ Training set: {X_train.shape[0]:,} samples ({y_train.mean()*100:.2f}% fraud)")
    print(f"   ✅ Validation set: {X_val.shape[0]:,} samples ({y_val.mean()*100:.2f}% fraud)")
    print(f"   ✅ Test set: {X_test.shape[0]:,} samples ({y_test.mean()*100:.2f}% fraud)")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    
    # Fit scaler only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("   ✅ Features scaled using StandardScaler")
    
    # Save processed data
    print("\n5. Saving processed data...")
    processed_dir = Path('../../data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_scaled.to_csv(processed_dir / 'X_train.csv', index=False)
    X_val_scaled.to_csv(processed_dir / 'X_val.csv', index=False)
    X_test_scaled.to_csv(processed_dir / 'X_test.csv', index=False)
    y_train.to_csv(processed_dir / 'y_train.csv', index=False)
    y_val.to_csv(processed_dir / 'y_val.csv', index=False)
    y_test.to_csv(processed_dir / 'y_test.csv', index=False)
    
    # Save scaler for production use
    with open(processed_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ Processed data saved to {processed_dir}/")
    print(f"   ✅ Scaler saved to {processed_dir}/scaler.pkl")
    
    # Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETED")
    print("="*60)
    print(f"Training samples: {len(X_train_scaled):,}")
    print(f"Validation samples: {len(X_val_scaled):,}")
    print(f"Test samples: {len(X_test_scaled):,}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print("="*60 + "\n")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    # Prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_prepare_data()
    
    print("✅ Data preparation complete!")
    print("   You can now proceed to model training.")