from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Optional
import os
import redis

# FastAPI app initialization
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system",
    version="1.0.0"
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mlops_user:mlops_password@localhost:5432/fraud_detection")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for request/response
class TransactionCreate(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    time: Optional[float] = Field(None, description="Time feature from dataset")
    v1: Optional[float] = None
    v2: Optional[float] = None
    v3: Optional[float] = None
    v4: Optional[float] = None
    v5: Optional[float] = None
    v6: Optional[float] = None
    v7: Optional[float] = None
    v8: Optional[float] = None
    v9: Optional[float] = None
    v10: Optional[float] = None
    v11: Optional[float] = None
    v12: Optional[float] = None
    v13: Optional[float] = None
    v14: Optional[float] = None
    v15: Optional[float] = None
    v16: Optional[float] = None
    v17: Optional[float] = None
    v18: Optional[float] = None
    v19: Optional[float] = None
    v20: Optional[float] = None
    v21: Optional[float] = None
    v22: Optional[float] = None
    v23: Optional[float] = None
    v24: Optional[float] = None
    v25: Optional[float] = None
    v26: Optional[float] = None
    v27: Optional[float] = None
    v28: Optional[float] = None
    is_fraud: Optional[int] = Field(None, description="Fraud label (0 or 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_001",
                "amount": 149.99,
                "time": 0.0,
                "v1": -1.359807,
                "is_fraud": 0
            }
        }

class TransactionResponse(BaseModel):
    id: int
    transaction_id: str
    status: str
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    timestamp: datetime

class StatsResponse(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_rate: float
    last_24h_transactions: int

# Routes
@app.get("/", response_model=dict)
def read_root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    
    # Check database
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        database=db_status,
        redis=redis_status,
        timestamp=datetime.utcnow()
    )

@app.post("/transactions/", response_model=TransactionResponse)
def create_transaction(transaction: TransactionCreate, db: Session = Depends(get_db)):
    """Create a new transaction"""
    
    try:
        # Build SQL insert query dynamically
        columns = ['transaction_id', 'amount', 'class']
        values = [transaction.transaction_id, transaction.amount, transaction.is_fraud or 0]
        
        # Add optional fields
        optional_fields = ['time'] + [f'v{i}' for i in range(1, 29)]
        for field in optional_fields:
            value = getattr(transaction, field, None)
            if value is not None:
                columns.append(field)
                values.append(value)
        
        # Create SQL query
        columns_str = ', '.join(columns)
        placeholders = ', '.join([':' + col for col in columns])
        query = text(f"INSERT INTO transactions ({columns_str}) VALUES ({placeholders}) RETURNING id, transaction_id, timestamp")
        
        # Create params dict
        params = {col: val for col, val in zip(columns, values)}
        
        # Execute query
        result = db.execute(query, params)
        db.commit()
        row = result.fetchone()
        
        # Cache in Redis
        redis_key = f"tx:{transaction.transaction_id}"
        redis_client.setex(redis_key, 3600, str(transaction.dict()))  # Cache for 1 hour
        
        return TransactionResponse(
            id=row[0],
            transaction_id=row[1],
            status="created",
            timestamp=row[2]
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating transaction: {str(e)}")

@app.get("/transactions/count")
def get_transaction_count(db: Session = Depends(get_db)):
    """Get total transaction count"""
    
    try:
        result = db.execute(text("SELECT COUNT(*) as count FROM transactions"))
        count = result.fetchone()[0]
        return {"total_transactions": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/stats", response_model=StatsResponse)
def get_transaction_stats(db: Session = Depends(get_db)):
    """Get transaction statistics"""
    
    try:
        # Total stats
        total_query = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) as fraud_count
            FROM transactions
        """)
        result = db.execute(total_query)
        row = result.fetchone()
        total = row[0]
        fraud = row[1] or 0
        fraud_rate = (fraud / total * 100) if total > 0 else 0
        
        # Last 24h stats
        last_24h_query = text("""
            SELECT COUNT(*) 
            FROM transactions 
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        """)
        result = db.execute(last_24h_query)
        last_24h = result.fetchone()[0]
        
        return StatsResponse(
            total_transactions=total,
            fraud_transactions=fraud,
            fraud_rate=round(fraud_rate, 2),
            last_24h_transactions=last_24h
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """Get a specific transaction by ID"""
    
    # Try Redis cache first
    redis_key = f"tx:{transaction_id}"
    cached = redis_client.get(redis_key)
    
    if cached:
        return {"source": "cache", "transaction": cached}
    
    # Query database
    try:
        query = text("SELECT * FROM transactions WHERE transaction_id = :tx_id")
        result = db.execute(query, {"tx_id": transaction_id})
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return {"source": "database", "transaction": dict(row._mapping)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)