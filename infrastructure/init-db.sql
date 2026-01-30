-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    time FLOAT,
    v1 FLOAT,
    v2 FLOAT,
    v3 FLOAT,
    v4 FLOAT,
    v5 FLOAT,
    v6 FLOAT,
    v7 FLOAT,
    v8 FLOAT,
    v9 FLOAT,
    v10 FLOAT,
    v11 FLOAT,
    v12 FLOAT,
    v13 FLOAT,
    v14 FLOAT,
    v15 FLOAT,
    v16 FLOAT,
    v17 FLOAT,
    v18 FLOAT,
    v19 FLOAT,
    v20 FLOAT,
    v21 FLOAT,
    v22 FLOAT,
    v23 FLOAT,
    v24 FLOAT,
    v25 FLOAT,
    v26 FLOAT,
    v27 FLOAT,
    v28 FLOAT,
    amount FLOAT,
    class INTEGER
);

-- Create index on timestamp for faster queries
CREATE INDEX idx_transactions_timestamp ON transactions (timestamp);

CREATE INDEX idx_transactions_class ON transactions (class);

-- Create a summary table for monitoring
CREATE TABLE IF NOT EXISTS transaction_stats (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_transactions INTEGER,
    fraud_transactions INTEGER,
    fraud_rate FLOAT,
    avg_amount FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);