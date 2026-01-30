"""
Advanced model training with multiple algorithms and sampling strategies.
Trains XGBoost and LightGBM with different class imbalance handling techniques.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed training and test data"""
    processed_dir = Path('../../data/processed')
    
    X_train = pd.read_csv(processed_dir / 'X_train.csv')
    X_val = pd.read_csv(processed_dir / 'X_val.csv')
    X_test = pd.read_csv(processed_dir / 'X_test.csv')
    y_train = pd.read_csv(processed_dir / 'y_train.csv').values.ravel()
    y_val = pd.read_csv(processed_dir / 'y_val.csv').values.ravel()
    y_test = pd.read_csv(processed_dir / 'y_test.csv').values.ravel()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost_smote(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train XGBoost with SMOTE oversampling"""
    
    mlflow.set_experiment("fraud-detection-advanced")
    
    with mlflow.start_run(run_name="xgboost_smote"):
        
        print("\n" + "="*60)
        print("TRAINING XGBOOST WITH SMOTE")
        print("="*60)
        
        # Parameters
        params = {
            "model_type": "xgboost",
            "sampling_strategy": "SMOTE",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # Apply SMOTE
        print("\n🔄 Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"   Original size: {len(y_train):,} (fraud: {y_train.sum():,})")
        print(f"   Resampled size: {len(y_train_resampled):,} (fraud: {y_train_resampled.sum():,})")
        print(f"   Fraud rate after SMOTE: {y_train_resampled.mean()*100:.2f}%")
        
        # Train model
        print("\n🔧 Training XGBoost...")
        start_time = time.time()
        
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=params['random_state'],
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)
        
        # Log metrics
        metrics = {
            "val_auc_roc": roc_auc_score(y_val, y_val_pred_proba),
            "test_auc_roc": roc_auc_score(y_test, y_test_pred_proba),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1_score": f1_score(y_test, y_test_pred),
        }
        mlflow.log_metrics(metrics)
        
        print(f"\n📊 Results:")
        print(f"   Val AUC: {metrics['val_auc_roc']:.4f}")
        print(f"   Test AUC: {metrics['test_auc_roc']:.4f}")
        print(f"   Precision: {metrics['test_precision']:.4f}")
        print(f"   Recall: {metrics['test_recall']:.4f}")
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        return model, metrics['test_auc_roc']

def train_xgboost_undersample(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train XGBoost with random undersampling"""
    
    mlflow.set_experiment("fraud-detection-advanced")
    
    with mlflow.start_run(run_name="xgboost_undersample"):
        
        print("\n" + "="*60)
        print("TRAINING XGBOOST WITH UNDERSAMPLING")
        print("="*60)
        
        params = {
            "model_type": "xgboost",
            "sampling_strategy": "undersample",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # Apply undersampling
        print("\n🔄 Applying Random Undersampling...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
        
        print(f"   Original size: {len(y_train):,}")
        print(f"   Undersampled size: {len(y_train_resampled):,}")
        print(f"   Fraud rate: {y_train_resampled.mean()*100:.2f}%")
        
        # Train
        print("\n🔧 Training XGBoost...")
        start_time = time.time()
        
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=params['random_state'],
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)
        
        metrics = {
            "val_auc_roc": roc_auc_score(y_val, y_val_pred_proba),
            "test_auc_roc": roc_auc_score(y_test, y_test_pred_proba),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1_score": f1_score(y_test, y_test_pred),
        }
        mlflow.log_metrics(metrics)
        
        print(f"\n📊 Results:")
        print(f"   Val AUC: {metrics['val_auc_roc']:.4f}")
        print(f"   Test AUC: {metrics['test_auc_roc']:.4f}")
        
        mlflow.xgboost.log_model(model, "model")
        
        return model, metrics['test_auc_roc']

def train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train LightGBM with scale_pos_weight"""
    
    mlflow.set_experiment("fraud-detection-advanced")
    
    with mlflow.start_run(run_name="lightgbm_scale_pos_weight"):
        
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM")
        print("="*60)
        
        # Calculate scale_pos_weight
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        params = {
            "model_type": "lightgbm",
            "sampling_strategy": "scale_pos_weight",
            "scale_pos_weight": float(scale_pos_weight),
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        print(f"\n⚖️  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train
        print("\n🔧 Training LightGBM...")
        start_time = time.time()
        
        model = LGBMClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            scale_pos_weight=scale_pos_weight,
            random_state=params['random_state'],
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)
        
        metrics = {
            "val_auc_roc": roc_auc_score(y_val, y_val_pred_proba),
            "test_auc_roc": roc_auc_score(y_test, y_test_pred_proba),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1_score": f1_score(y_test, y_test_pred),
        }
        mlflow.log_metrics(metrics)
        
        print(f"\n📊 Results:")
        print(f"   Val AUC: {metrics['val_auc_roc']:.4f}")
        print(f"   Test AUC: {metrics['test_auc_roc']:.4f}")
        
        mlflow.lightgbm.log_model(model, "model")
        
        return model, metrics['test_auc_roc']

def train_xgboost_class_weight(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train XGBoost with no resampling (using scale_pos_weight)"""
    
    mlflow.set_experiment("fraud-detection-advanced")
    
    with mlflow.start_run(run_name="xgboost_class_weight"):
        
        print("\n" + "="*60)
        print("TRAINING XGBOOST (Class Weight)")
        print("="*60)
        
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        params = {
            "model_type": "xgboost",
            "sampling_strategy": "class_weight",
            "scale_pos_weight": float(scale_pos_weight),
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # Train
        print(f"\n⚖️  Scale pos weight: {scale_pos_weight:.2f}")
        print("\n🔧 Training XGBoost...")
        start_time = time.time()
        
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            scale_pos_weight=scale_pos_weight,
            random_state=params['random_state'],
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)
        
        metrics = {
            "val_auc_roc": roc_auc_score(y_val, y_val_pred_proba),
            "test_auc_roc": roc_auc_score(y_test, y_test_pred_proba),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1_score": f1_score(y_test, y_test_pred),
        }
        mlflow.log_metrics(metrics)
        
        print(f"\n📊 Results:")
        print(f"   Val AUC: {metrics['val_auc_roc']:.4f}")
        print(f"   Test AUC: {metrics['test_auc_roc']:.4f}")
        
        mlflow.xgboost.log_model(model, "model")
        
        return model, metrics['test_auc_roc']


if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Load data
    print("Loading processed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Store results
    results = {}
    
    # Train different models
    print("\n" + "="*60)
    print("TRAINING MULTIPLE MODELS")
    print("="*60)
    
    # 1. XGBoost with SMOTE
    _, auc1 = train_xgboost_smote(X_train, X_val, X_test, y_train, y_val, y_test)
    results['XGBoost_SMOTE'] = auc1
    
    # 2. XGBoost with undersampling
    _, auc2 = train_xgboost_undersample(X_train, X_val, X_test, y_train, y_val, y_test)
    results['XGBoost_Undersample'] = auc2
    
    # 3. XGBoost with class weight
    _, auc3 = train_xgboost_class_weight(X_train, X_val, X_test, y_train, y_val, y_test)
    results['XGBoost_ClassWeight'] = auc3
    
    # 4. LightGBM
    _, auc4 = train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test)
    results['LightGBM'] = auc4
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:30s}: AUC = {auc:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\n🏆 Best model: {best_model} (AUC: {results[best_model]:.4f})")
    print(f"\n🌐 View all experiments at: http://localhost:5000")