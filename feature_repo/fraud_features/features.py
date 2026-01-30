"""
Feature definitions for fraud detection system.
Defines entities, feature views, and online/offline stores configuration.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int64, String, UnixTimestamp

# Define the user entity
user = Entity(
    name="user_id",
    description="User ID for credit card transactions",
    value_type=ValueType.STRING,
)

# Define data source - pointing to our enhanced dataset
path = "/home/jcarloshg/Documents/school/RealTimeFraudDetectionSystem-MLOps/data/processed/enhanced_transactions.parquet"
transaction_source = FileSource(
    name="transaction_source",
    path=path,
    timestamp_field="event_timestamp",
)

# Feature View 1: Transaction Statistics (PCA features V1-V28)
transaction_features = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Field(name="V1", dtype=Float64),
        Field(name="V2", dtype=Float64),
        Field(name="V3", dtype=Float64),
        Field(name="V4", dtype=Float64),
        Field(name="V5", dtype=Float64),
        Field(name="V6", dtype=Float64),
        Field(name="V7", dtype=Float64),
        Field(name="V8", dtype=Float64),
        Field(name="V9", dtype=Float64),
        Field(name="V10", dtype=Float64),
        Field(name="V11", dtype=Float64),
        Field(name="V12", dtype=Float64),
        Field(name="V13", dtype=Float64),
        Field(name="V14", dtype=Float64),
        Field(name="V15", dtype=Float64),
        Field(name="V16", dtype=Float64),
        Field(name="V17", dtype=Float64),
        Field(name="V18", dtype=Float64),
        Field(name="V19", dtype=Float64),
        Field(name="V20", dtype=Float64),
        Field(name="V21", dtype=Float64),
        Field(name="V22", dtype=Float64),
        Field(name="V23", dtype=Float64),
        Field(name="V24", dtype=Float64),
        Field(name="V25", dtype=Float64),
        Field(name="V26", dtype=Float64),
        Field(name="V27", dtype=Float64),
        Field(name="V28", dtype=Float64),
        Field(name="Amount", dtype=Float64),
    ],
    source=transaction_source,
    online=True,
)

# Feature View 2: Temporal features
temporal_features = FeatureView(
    name="temporal_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
    ],
    source=transaction_source,
    online=True,
)