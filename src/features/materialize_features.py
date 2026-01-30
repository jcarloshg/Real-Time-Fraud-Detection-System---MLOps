"""
Materialize features from offline to online store (Redis).
This makes features available for real-time inference.
"""

from feast import FeatureStore
from datetime import datetime, timedelta
import sys
from pathlib import Path

def materialize_features(start_date=None, end_date=None):
    """
    Materialize features to the online store
    
    Args:
        start_date: Start of materialization window
        end_date: End of materialization window
    """
    
    print("="*60)
    print("FEATURE MATERIALIZATION")
    print("="*60)
    
    # Initialize Feast
    print("\n1. Initializing Feast FeatureStore...")
    
    # Update path to your feature_repo
    repo_path = Path(__file__).parent.parent.parent / "feature_repo" / "fraud_features"
    print(f"repo_path {repo_path}")
    print(f"   Feature repo path: {repo_path}")
    
    try:
        store = FeatureStore(repo_path=str(repo_path))
        print("   ✅ FeatureStore initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize FeatureStore: {e}")
        print("\n   Make sure you've run 'feast apply' in the feature_repo/fraud_features directory")
        return False
    
    # Set materialization window
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)  # Last 30 days
    
    print(f"\n2. Materialization window:")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    
    # Materialize features
    print("\n3. Materializing features to online store (Redis)...")
    try:
        store.materialize(
            start_date=start_date,
            end_date=end_date
        )
        print("   ✅ Features materialized successfully!")
        
        # Get feature views
        feature_views = store.list_feature_views()
        print(f"\n   Materialized feature views:")
        for fv in feature_views:
            print(f"   - {fv.name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Materialization failed: {e}")
        return False

if __name__ == "__main__":
    # Materialize last 90 days of data
    end_date = datetime(2024, 1, 3)  # Adjust based on your data
    start_date = datetime(2024, 1, 1)
    
    success = materialize_features(start_date, end_date)
    
    if success:
        print("\n" + "="*60)
        print("✅ MATERIALIZATION COMPLETE")
        print("="*60)
        print("\nFeatures are now available in Redis for online serving!")
    else:
        print("\n❌ Materialization failed. Please check the errors above.")
        sys.exit(1)