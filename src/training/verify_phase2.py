"""
Verification script for Phase 2 completion
"""

import mlflow
from mlflow.tracking import MlflowClient
import sys

def verify_phase2():
    """Verify all Phase 2 components"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    print("="*60)
    print("PHASE 2 VERIFICATION")
    print("="*60)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: MLflow connection
    print("\n1. Checking MLflow connection...")
    try:
        experiments = client.search_experiments()
        print(f"   ✅ Connected to MLflow ({len(experiments)} experiments found)")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Failed to connect to MLflow: {e}")
    
    # Check 2: Baseline experiment exists
    print("\n2. Checking baseline experiment...")
    exp_baseline = client.get_experiment_by_name("fraud-detection-baseline")
    if exp_baseline:
        runs = client.search_runs([exp_baseline.experiment_id])
        print(f"   ✅ Baseline experiment found ({len(runs)} runs)")
        checks_passed += 1
    else:
        print("   ❌ Baseline experiment not found")
    
    # Check 3: Advanced experiment exists
    print("\n3. Checking advanced experiment...")
    exp_advanced = client.get_experiment_by_name("fraud-detection-advanced")
    if exp_advanced:
        runs = client.search_runs([exp_advanced.experiment_id])
        print(f"   ✅ Advanced experiment found ({len(runs)} runs)")
        if len(runs) >= 3:
            print(f"   ✅ Multiple models trained")
            checks_passed += 1
        else:
            print(f"   ⚠️  Only {len(runs)} runs found (expected 3+)")
    else:
        print("   ❌ Advanced experiment not found")
    
    # Check 4: Registered model
    print("\n4. Checking registered model...")
    try:
        registered_models = client.search_registered_models("name='fraud-detection-model'")
        if registered_models:
            model = registered_models[0]
            print(f"   ✅ Model registered: {model.name}")
            print(f"   ✅ Versions: {len(model.latest_versions)}")
            checks_passed += 1
        else:
            print("   ❌ No registered model found")
    except Exception as e:
        print(f"   ❌ Error checking registered model: {e}")
    
    # Check 5: Processed data exists
    print("\n5. Checking processed data...")
    from pathlib import Path
    processed_dir = Path('../../data/processed')
    required_files = ['X_train.csv', 'X_val.csv', 'X_test.csv', 
                     'y_train.csv', 'y_val.csv', 'y_test.csv', 'scaler.pkl']
    
    all_exist = all((processed_dir / f).exists() for f in required_files)
    if all_exist:
        print(f"   ✅ All processed data files exist")
        checks_passed += 1
    else:
        print(f"   ❌ Some processed data files missing")
        for f in required_files:
            status = "✅" if (processed_dir / f).exists() else "❌"
            print(f"      {status} {f}")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("✅ Phase 2 COMPLETE! All checks passed.")
        print("\nYou can now proceed to Phase 3: Feature Engineering")
        return True
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = verify_phase2()
    sys.exit(0 if success else 1)