"""
Register the best performing model to MLflow Model Registry.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

def get_best_run(experiment_name, metric="test_auc_roc"):
    """
    Find the best run from an experiment based on a metric
    """
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found!")
        return None
    
    print(f"\n🔍 Searching for best run in experiment: {experiment_name}")
    
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} DESC"],
        max_results=10
    )
    
    if not runs:
        print("❌ No runs found in the experiment!")
        return None
    
    # Display top 5 runs
    print(f"\n📊 Top 5 runs by {metric}:")
    print("-" * 80)
    for i, run in enumerate(runs[:5], 1):
        model_type = run.data.params.get('model_type', 'unknown')
        sampling = run.data.params.get('sampling_strategy', 'none')
        metric_value = run.data.metrics.get(metric, 0)
        print(f"{i}. {model_type:15s} | {sampling:20s} | {metric}: {metric_value:.4f}")
    
    best_run = runs[0]
    return best_run

def register_best_model(experiment_name, model_name="fraud-detection-model", 
                       metric="test_auc_roc"):
    """
    Register the best model from an experiment
    """
    client = MlflowClient()
    
    # Get best run
    best_run = get_best_run(experiment_name, metric)
    
    if best_run is None:
        return
    
    # Extract run details
    run_id = best_run.info.run_id
    model_type = best_run.data.params.get('model_type', 'unknown')
    sampling = best_run.data.params.get('sampling_strategy', 'none')
    auc = best_run.data.metrics.get(metric, 0)
    
    print("\n" + "="*80)
    print("BEST MODEL FOUND")
    print("="*80)
    print(f"Run ID: {run_id}")
    print(f"Model Type: {model_type}")
    print(f"Sampling Strategy: {sampling}")
    print(f"{metric}: {auc:.4f}")
    print("="*80)
    
    # Check if model is already registered
    try:
        registered_models = client.search_registered_models(f"name='{model_name}'")
        if registered_models:
            print(f"\n📋 Model '{model_name}' already exists")
            print(f"   Existing versions: {len(registered_models[0].latest_versions)}")
    except Exception as e:
        print(f"\n📝 Model '{model_name}' does not exist yet")
    
    # Register the model
    model_uri = f"runs:/{run_id}/model"
    
    try:
        print(f"\n📦 Registering model...")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"✅ Model registered successfully!")
        print(f"   Name: {model_name}")
        print(f"   Version: {registered_model.version}")
        
        # Add description
        description = f"""
        Fraud Detection Model
        - Model Type: {model_type}
        - Sampling Strategy: {sampling}
        - Test AUC-ROC: {auc:.4f}
        - Trained on: {best_run.info.start_time}
        """
        
        client.update_model_version(
            name=model_name,
            version=registered_model.version,
            description=description.strip()
        )
        
        print(f"\n📝 Description added")
        
        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )
        
        print(f"✅ Model transitioned to 'Staging' stage")
        
        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="model_type",
            value=model_type
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="sampling_strategy",
            value=sampling
        )
        
        print(f"✅ Tags added")
        
        print("\n" + "="*80)
        print("REGISTRATION COMPLETE")
        print("="*80)
        print(f"🌐 View registered model at:")
        print(f"   http://localhost:5000/#/models/{model_name}")
        print("="*80)
        
        return registered_model
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return None

def promote_to_production(model_name, version=None):
    """
    Promote a model version to Production stage
    """
    client = MlflowClient()
    
    if version is None:
        # Get latest version in Staging
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not versions:
            print("❌ No models in Staging stage")
            return
        version = versions[0].version
    
    print(f"\n📦 Promoting model '{model_name}' version {version} to Production...")
    
    # Archive current production models
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=pv.version,
            stage="Archived"
        )
        print(f"   📥 Archived version {pv.version}")
    
    # Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"✅ Version {version} promoted to Production!")


if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    print("="*80)
    print("MLflow MODEL REGISTRY")
    print("="*80)
    
    # Register best model from advanced experiments
    registered_model = register_best_model(
        experiment_name="fraud-detection-advanced",
        model_name="fraud-detection-model",
        metric="test_auc_roc"
    )
    
    if registered_model:
        # Optionally promote to production
        print("\n" + "="*80)
        response = input("Do you want to promote this model to Production? (y/n): ")
        if response.lower() == 'y':
            promote_to_production("fraud-detection-model", registered_model.version)