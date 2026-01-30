"""
Baseline model training with MLflow tracking.
Trains Logistic Regression with class imbalance handling.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed training and test data"""
    # Define data directory
    processed_dir = Path('../../data/processed')
    
    print("Loading processed data...")
    X_train = pd.read_csv(processed_dir / 'X_train.csv')
    X_val = pd.read_csv(processed_dir / 'X_val.csv')
    X_test = pd.read_csv(processed_dir / 'X_test.csv')
    y_train = pd.read_csv(processed_dir / 'y_train.csv').values.ravel()
    y_val = pd.read_csv(processed_dir / 'y_val.csv').values.ravel()
    y_test = pd.read_csv(processed_dir / 'y_test.csv').values.ravel()
    
    print(f"✅ Data loaded:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add labels
    plt.gca().xaxis.set_ticklabels(['Normal', 'Fraud'])
    plt.gca().yaxis.set_ticklabels(['Normal', 'Fraud'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def plot_roc_curve(y_true, y_pred_proba, save_path='roc_curve.png'):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def plot_precision_recall_curve(y_true, y_pred_proba, save_path='pr_curve.png'):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test, 
                              class_weight='balanced', max_iter=1000):
    """
    Train Logistic Regression with MLflow tracking
    """
    
    # Set experiment
    mlflow.set_experiment("fraud-detection-baseline")
    
    with mlflow.start_run(run_name="logistic_regression_balanced"):
        
        print("="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        # Log parameters
        params = {
            "model_type": "logistic_regression",
            "class_weight": class_weight,
            "max_iter": max_iter,
            "solver": "lbfgs",
            "random_state": 42
        }
        
        mlflow.log_params(params)
        print("\n📋 Parameters logged to MLflow")
        
        # Train model
        print("\n🔧 Training model...")
        start_time = time.time()
        
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        print(f"   ✅ Model trained in {training_time:.2f} seconds")
        
        # Predictions on validation set
        print("\n📊 Evaluating on validation set...")
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Validation metrics
        val_metrics = {
            "val_auc_roc": roc_auc_score(y_val, y_val_pred_proba),
            "val_precision": precision_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
            "val_f1_score": f1_score(y_val, y_val_pred),
            "val_avg_precision": average_precision_score(y_val, y_val_pred_proba)
        }
        
        mlflow.log_metrics(val_metrics)
        
        print("\n   Validation Metrics:")
        print(f"   AUC-ROC: {val_metrics['val_auc_roc']:.4f}")
        print(f"   Precision: {val_metrics['val_precision']:.4f}")
        print(f"   Recall: {val_metrics['val_recall']:.4f}")
        print(f"   F1-Score: {val_metrics['val_f1_score']:.4f}")
        print(f"   Avg Precision: {val_metrics['val_avg_precision']:.4f}")
        
        # Predictions on test set
        print("\n📊 Evaluating on test set...")
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Test metrics
        test_metrics = {
            "test_auc_roc": roc_auc_score(y_test, y_test_pred_proba),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1_score": f1_score(y_test, y_test_pred),
            "test_avg_precision": average_precision_score(y_test, y_test_pred_proba)
        }
        
        mlflow.log_metrics(test_metrics)
        
        print("\n   Test Metrics:")
        print(f"   AUC-ROC: {test_metrics['test_auc_roc']:.4f}")
        print(f"   Precision: {test_metrics['test_precision']:.4f}")
        print(f"   Recall: {test_metrics['test_recall']:.4f}")
        print(f"   F1-Score: {test_metrics['test_f1_score']:.4f}")
        print(f"   Avg Precision: {test_metrics['test_avg_precision']:.4f}")
        
        # Create and log plots
        print("\n📈 Creating visualizations...")
        cm_path = plot_confusion_matrix(y_test, y_test_pred, 'lr_confusion_matrix.png')
        roc_path = plot_roc_curve(y_test, y_test_pred_proba, 'lr_roc_curve.png')
        pr_path = plot_precision_recall_curve(y_test, y_test_pred_proba, 'lr_pr_curve.png')
        
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        print("   ✅ Plots saved and logged to MLflow")
        
        # Log model
        print("\n💾 Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=None  # We'll register later
        )
        print("   ✅ Model logged successfully")
        
        # Print classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT (Test Set)")
        print("="*60)
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Normal', 'Fraud']))
        print("="*60)
        
        return model, test_metrics['test_auc_roc']


if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Train baseline model
    model, auc = train_logistic_regression(X_train, X_val, X_test, 
                                           y_train, y_val, y_test)
    
    print(f"\n✅ Training completed!")
    print(f"   Best Test AUC: {auc:.4f}")
    print(f"\n🌐 View results in MLflow UI: http://localhost:5000")