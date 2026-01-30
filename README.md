## steps for PHASE 2

### 1.2 Create MLflow database in PostgreSQL

```bash
# Start PostgreSQL if not already running
sudo docker compose up -d postgres

# Wait a few seconds for PostgreSQL to be ready, then create MLflow database
sudo docker exec -it fraud_postgres psql -U mlops_user -d postgres -c "CREATE DATABASE mlflow_db;"

# Verify the database was created
sudo docker exec -it fraud_postgres psql -U mlops_user -d postgres -c "\l"
```

### 1.3 Start MLflow

```bash
# Start MLflow service
sudo docker compose up -d mlflow

# Check if it's running
sudo docker compose ps mlflow

# View logs to ensure no errors
sudo docker compose logs -f mlflow
```

## Steps for PHASE 3

### 1.5 Create Feast registry database:

```bash
# Create database for Feast registry
docker exec -it fraud_postgres psql -U mlops_user -d postgres -c "CREATE DATABASE feast_registry;"

# Verify
docker exec -it fraud_postgres psql -U mlops_user -d postgres -c "\l" | grep feast
```