"""
Compute and materialize features for offline training and online serving.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from feast import FeatureStore

def compute_user_aggregates(df):
    """
    Compute user-level aggregated features
    
    Args:
        df: DataFrame with enhanced transactions
    
    Returns:
        DataFrame with user-level features
    """
    
    print("\n📊 Computing user-level aggregated features...")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Features per user
    user_features = df.groupby('user_id').agg({
        'Amount': ['mean', 'std', 'min', 'max', 'count'],
        'fraud_label': 'sum',
        'timestamp': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    user_features.columns = ['_'.join(col).strip('_') for col in user_features.columns.values]
    user_features = user_features.rename(columns={'user_id': 'user_id'})
    
    # Add derived features
    user_features['fraud_rate'] = user_features['fraud_label_sum'] / user_features['Amount_count']
    user_features['days_active'] = (user_features['timestamp_max'] - user_features['timestamp_min']).dt.total_seconds() / 86400
    
    print(f"   ✅ Computed features for {len(user_features):,} users")
    
    return user_features

def compute_temporal_aggregates(df, windows=['1h', '24h', '7d']):
    """
    Compute rolling time-window features
    
    Args:
        df: DataFrame with enhanced transactions
        windows: List of time windows for aggregation
    
    Returns:
        DataFrame with temporal features
    """
    
    print(f"\n⏱️  Computing temporal features for windows: {windows}...")
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    temporal_features = []
    
    for user_id, user_df in df.groupby('user_id'):
        user_temporal = pd.DataFrame()
        user_temporal['user_id'] = [user_id] * len(user_df)
        user_temporal['timestamp'] = user_df.index
        
        # Compute rolling features for each window
        for window in windows:
            # Transaction count
            user_temporal[f'tx_count_{window}'] = user_df['Amount'].rolling(window).count()
            
            # Amount statistics
            user_temporal[f'amount_mean_{window}'] = user_df['Amount'].rolling(window).mean()
            user_temporal[f'amount_std_{window}'] = user_df['Amount'].rolling(window).std()
            user_temporal[f'amount_max_{window}'] = user_df['Amount'].rolling(window).max()
        
        temporal_features.append(user_temporal)
    
    result = pd.concat(temporal_features, ignore_index=True)
    result = result.fillna(0)  # Fill NaN for early transactions
    
    print(f"   ✅ Computed temporal features for {len(result):,} transactions")
    
    return result

def save_feature_data(df, output_path='../../data/processed/feature_data.parquet'):
    """
    Save feature data in parquet format for efficient loading
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"\n💾 Features saved to {output_path}")

def main():
    """
    Main feature computation pipeline
    """
    
    print("="*60)
    print("FEATURE COMPUTATION PIPELINE")
    print("="*60)
    
    # Load enhanced dataset
    print("\n1. Loading enhanced dataset...")
    df = pd.read_csv('../../data/processed/enhanced_transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    print(f"   ✅ Loaded {len(df):,} transactions")
    
    # Compute user aggregates
    print("\n2. Computing user-level features...")
    user_features = compute_user_aggregates(df)
    user_features.to_csv('../../data/processed/user_features.csv', index=False)
    print("   ✅ User features saved")
    
    # Compute temporal aggregates (using smaller windows for demonstration)
    print("\n3. Computing temporal features...")
    temporal_features = compute_temporal_aggregates(df.head(10000), windows=['1h', '24h'])
    temporal_features.to_csv('../../data/processed/temporal_features.csv', index=False)
    print("   ✅ Temporal features saved")
    
    print("\n" + "="*60)
    print("FEATURE COMPUTATION COMPLETED")
    print("="*60)
    print(f"\nUser features shape: {user_features.shape}")
    print(f"Temporal features shape: {temporal_features.shape}")
    print("\n✅ All features computed successfully!")

if __name__ == "__main__":
    main()