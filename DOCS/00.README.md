# Self-Hosted MLOps Project: Real-Time Fraud Detection System

This is an excellent learning project because it touches every phase of the MLOps lifecycle while remaining achievable on modest hardware.

---

## Why Fraud Detection?

This use case is ideal for learning MLOps because it involves:

- **Streaming and batch processing** — you'll handle both real-time predictions and periodic retraining
- **Class imbalance** — a real-world data challenge you'll need to address
- **Concept drift** — fraud patterns evolve, so your model will degrade without monitoring
- **Low-latency requirements** — forces you to think about serving infrastructure
- **Clear feedback loops** — transactions are eventually labeled as fraudulent or legitimate

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Your Home Lab                                     │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Kafka /    │───▶│   Feature    │───▶│    Model     │                  │
│  │   Redis      │    │   Pipeline   │    │   Serving    │                  │
│  │  (events)    │    │   (Feast)    │    │  (FastAPI)   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  PostgreSQL  │    │    MLflow    │    │  Prometheus  │                  │
│  │  (storage)   │    │  (tracking)  │    │  + Grafana   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                             │                                               │
│                             ▼                                               │
│                      ┌──────────────┐                                       │
│                      │   Airflow    │                                       │
│                      │ (orchestrate)│                                       │
│                      └──────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

This can run on surprisingly modest hardware:

| Component | Minimum                    | Recommended                    |
| --------- | -------------------------- | ------------------------------ |
| CPU       | 4 cores                    | 8 cores                        |
| RAM       | 16 GB                      | 32 GB                          |
| Storage   | 100 GB SSD                 | 250 GB SSD                     |
| Setup     | Single machine with Docker | 2-3 Raspberry Pis or small VMs |

A single machine with Docker Compose works fine for learning. If you want to simulate a distributed environment, use Minikube or k3s.

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal:** Get your infrastructure running and ingest data.

**Tasks:**

1. Set up Docker Compose with PostgreSQL, Redis, and a basic FastAPI service
2. Download the Kaggle Credit Card Fraud dataset (or generate synthetic data with CTGAN)
3. Build a simple data ingestion script that simulates streaming transactions
4. Create a basic exploratory notebook to understand the data

**Tech stack for this phase:**

- Docker + Docker Compose
- PostgreSQL
- Python + pandas for exploration

**Deliverable:** A running system that ingests transactions into your database.

---

### Phase 2: Experiment Tracking & Initial Model (Week 3-4)

**Goal:** Train a baseline model with proper experiment tracking.

**Tasks:**

1. Set up MLflow server (with PostgreSQL backend and local artifact storage)
2. Build a training pipeline that logs parameters, metrics, and artifacts
3. Train a baseline model (start with logistic regression, then try XGBoost or LightGBM)
4. Handle class imbalance (SMOTE, class weights, or threshold tuning)
5. Register your best model in MLflow Model Registry

**Key learnings:**

- Experiment reproducibility
- Hyperparameter logging
- Model versioning

**Deliverable:** A trained model registered in MLflow with full experiment lineage.

---

### Phase 3: Feature Engineering & Feature Store (Week 5-6)

**Goal:** Build reusable features with a proper feature store.

**Tasks:**

1. Install and configure Feast with Redis as the online store
2. Define feature views for transaction features:
   - Transaction amount statistics (rolling mean, std over last 24h)
   - Transaction frequency per user
   - Distance from last transaction location
   - Time since last transaction
3. Build a batch pipeline to compute and materialize features
4. Update your training pipeline to pull features from Feast

**Key learnings:**

- Avoiding training/serving skew
- Feature versioning
- Online vs. offline feature stores

**Deliverable:** A working feature store that serves consistent features for training and inference.

---

### Phase 4: Model Serving (Week 7-8)

**Goal:** Deploy your model as a real-time API.

**Tasks:**

1. Build a FastAPI service that loads the model from MLflow
2. Implement feature retrieval from Feast at inference time
3. Add input validation with Pydantic
4. Containerize the service with Docker
5. Add basic health checks and readiness probes

**Example endpoint structure:**

```
POST /predict
{
  "transaction_id": "tx_123",
  "user_id": "user_456",
  "amount": 150.00,
  "merchant_category": "electronics",
  "timestamp": "2024-01-15T14:30:00Z"
}

Response:
{
  "transaction_id": "tx_123",
  "fraud_probability": 0.03,
  "is_fraud": false,
  "model_version": "1.2.0",
  "latency_ms": 12
}
```

**Deliverable:** A containerized prediction API that serves real-time fraud scores.

---

### Phase 5: Monitoring & Observability (Week 9-10)

**Goal:** Know when your model is healthy or degrading.

**Tasks:**

1. Set up Prometheus to scrape metrics from your FastAPI service
2. Configure Grafana dashboards for:
   - Request latency (P50, P95, P99)
   - Prediction distribution (% flagged as fraud)
   - Feature value distributions
   - Error rates
3. Add Evidently AI for drift detection:
   - Data drift on input features
   - Prediction drift
4. Configure alerts for anomalies (e.g., fraud rate spikes above 5%)

**Key metrics to track:**

| Metric                        | Why It Matters                           |
| ----------------------------- | ---------------------------------------- |
| Prediction latency            | User experience, SLA compliance          |
| Fraud rate                    | Sudden changes indicate drift or attacks |
| Feature null rates            | Data pipeline issues                     |
| Model confidence distribution | Shifts suggest concept drift             |

**Deliverable:** Dashboards showing model health and alerts for anomalies.

---

### Phase 6: Pipeline Orchestration & Retraining (Week 11-12)

**Goal:** Automate the full lifecycle from data to deployment.

**Tasks:**

1. Set up Airflow (or Dagster if you prefer)
2. Create DAGs for:
   - Daily feature materialization
   - Weekly model retraining
   - Model evaluation against production baseline
3. Implement a promotion gate: new model only deploys if it beats current production model
4. Add a simple CI pipeline (GitHub Actions) that runs tests on code changes

**Retraining trigger logic:**

```
if (performance_degradation > threshold) or (days_since_last_training > 7):
    trigger_retraining()

if new_model_auc > production_model_auc + min_improvement:
    promote_to_production()
else:
    alert_team_for_review()
```

**Deliverable:** Automated pipelines that retrain and potentially redeploy your model.

---

## Stretch Goals (If You Want More Challenge)

Once you have the core system running:

1. **Add Kubernetes** — Deploy on k3s or Minikube with proper resource limits and autoscaling
2. **Implement canary deployments** — Route 10% of traffic to new models before full rollout
3. **Build a labeling interface** — Simple UI to mark transactions as fraud/legitimate, feeding back into training
4. **Add explainability** — Integrate SHAP to explain individual predictions
5. **Simulate an attack** — Inject synthetic fraud patterns and watch your monitoring catch the drift

---

## Dataset Options

| Dataset                  | Pros                               | Cons                       |
| ------------------------ | ---------------------------------- | -------------------------- |
| Kaggle Credit Card Fraud | Real data, well-documented         | Small, static              |
| IEEE-CIS Fraud Detection | Larger, more features              | More complex preprocessing |
| Synthetic (CTGAN/Faker)  | Unlimited size, controllable drift | Not real patterns          |
| PaySim                   | Simulates mobile money             | Somewhat artificial        |

I'd recommend starting with Kaggle Credit Card Fraud for simplicity, then generating synthetic data when you want to simulate drift scenarios.

---

## Directory Structure Suggestion

```
fraud-detection-mlops/
├── docker-compose.yml
├── README.md
├── data/
│   └── raw/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   └── validation.py
│   ├── features/
│   │   ├── feature_definitions.py
│   │   └── feature_store.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── serving/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── Dockerfile
│   └── monitoring/
│       └── drift_detection.py
├── pipelines/
│   └── airflow/
│       └── dags/
├── infrastructure/
│   ├── prometheus/
│   └── grafana/
└── tests/
    ├── test_features.py
    └── test_serving.py
```

---

## What You'll Learn

By completing this project, you'll have hands-on experience with:

- Experiment tracking and model versioning
- Feature stores and avoiding training/serving skew
- Model serving with FastAPI
- Infrastructure monitoring with Prometheus/Grafana
- ML-specific monitoring (drift detection)
- Pipeline orchestration with Airflow
- Containerization and basic deployment

This gives you a solid foundation that translates directly to production MLOps work. Start simple, get each piece working, then add complexity. The goal is understanding, not perfection.
