"""
Create enhanced dataset with user IDs and temporal features
for feature engineering in Feast.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

def generate_user_id(row_index, total_users=10000):
    """Generate consistent user ID based on transaction patterns"""
    # Use modulo to assign users (fraudsters tend to have multiple transactions)
    user_num = row_index % total_users
    return f"user_{user_num:06d}"

def create_enhanced_dataset(input_path='../../data/raw/creditcard.csv',
                           output_path='../../data/processed/enhanced_transactions.csv'):
    """
    Create enhanced dataset with user IDs and timestamps
    """
    
    print("="*60)
    print("CREATING ENHANCED DATASET")
    print("="*60)
    
    # Load original data
    print(f"\n1. Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   ✅ Loaded {len(df):,} transactions")
    
    # Add user IDs
    print("\n2. Generating user IDs...")
    df['user_id'] = df.index.map(lambda x: generate_user_id(x, total_users=10000))
    print(f"   ✅ Assigned {df['user_id'].nunique():,} unique users")
    
    # Create actual timestamps from the Time column
    # Time is seconds from first transaction
    print("\n3. Creating timestamps...")
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    df['timestamp'] = df['Time'].apply(lambda x: base_time + timedelta(seconds=x))
    df['event_timestamp'] = df['timestamp']  # Feast requires event_timestamp
    
    print(f"   ✅ Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Add transaction ID
    print("\n4. Adding transaction IDs...")
    df['transaction_id'] = df.index.map(lambda x: f"tx_{x:08d}")
    
    # Rename Class to fraud_label for clarity
    df['fraud_label'] = df['Class']
    
    # Add hour of day and day of week features
    print("\n5. Adding temporal features...")
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Reorder columns
    cols_order = ['transaction_id', 'user_id', 'timestamp', 'event_timestamp', 
                  'Amount', 'fraud_label'] + \
                 [f'V{i}' for i in range(1, 29)] + \
                 ['hour_of_day', 'day_of_week', 'Time', 'Class']
    
    df = df[cols_order]
    
    # Save enhanced dataset
    print(f"\n6. Saving enhanced dataset...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED DATASET SUMMARY")
    print("="*60)
    print(f"Total transactions: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Fraud rate: {df['fraud_label'].mean()*100:.2f}%")
    print(f"Columns: {len(df.columns)}")
    print("="*60 + "\n")
    
    return df

if __name__ == "__main__":
    df = create_enhanced_dataset()
    print("✅ Enhanced dataset created successfully!")