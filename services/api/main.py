from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

app = FastAPI(title="Fraud Detection API")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Transaction model


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float)
    merchant_category = Column(String)
    is_fraud = Column(Integer, default=0)


# Create tables
Base.metadata.create_all(bind=engine)


class TransactionInput(BaseModel):
    amount: float
    merchant_category: str


@app.get("/")
def read_root():
    return {"status": "Fraud Detection API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/transactions/")
def create_transaction(transaction: TransactionInput):
    db = SessionLocal()
    try:
        db_transaction = Transaction(
            amount=transaction.amount,
            merchant_category=transaction.merchant_category
        )
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        return {"id": db_transaction.id, "status": "created"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/transactions/count")
def get_transaction_count():
    db = SessionLocal()
    try:
        count = db.query(Transaction).count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
