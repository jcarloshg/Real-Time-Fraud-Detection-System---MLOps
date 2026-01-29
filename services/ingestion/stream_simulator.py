import pandas as pd
import psycopg
from psycopg.extras import execute_values
import time
import random
from datetime import datetime
import os


class TransactionStreamer:
    def __init__(self, csv_path, db_config):
        self.df = pd.read_csv(csv_path)
        self.db_config = db_config
        self.current_index = 0

    def connect_db(self):
        return psycopg.connect(**self.db_config)

    def create_table(self):
        conn = self.connect_db()
        cur = conn.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            time FLOAT,
            v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT, v5 FLOAT,
            v6 FLOAT, v7 FLOAT, v8 FLOAT, v9 FLOAT, v10 FLOAT,
            v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT, v15 FLOAT,
            v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
            v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT,
            v26 FLOAT, v27 FLOAT, v28 FLOAT,
            amount FLOAT,
            class INTEGER
        );
        """
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        conn.close()
        print("Table created successfully")

    def stream_transactions(self, batch_size=10, delay=2):
        """Simulate streaming by inserting batches of transactions"""
        self.create_table()
        conn = self.connect_db()
        cur = conn.cursor()

        print(f"Starting to stream {len(self.df)} transactions...")

        while self.current_index < len(self.df):
            # Get batch
            batch = self.df.iloc[self.current_index:self.current_index + batch_size]

            # Prepare data for insertion
            values = []
            for _, row in batch.iterrows():
                values.append(tuple(row.values))

            # Insert batch
            columns = ','.join(self.df.columns)
            placeholders = ','.join(['%s'] * len(self.df.columns))
            insert_query = f"INSERT INTO transactions ({columns}) VALUES %s"

            execute_values(cur, insert_query, values)
            conn.commit()

            self.current_index += batch_size
            print(
                f"Inserted batch. Total: {self.current_index}/{len(self.df)}")

            # Simulate streaming delay
            time.sleep(delay)

        cur.close()
        conn.close()
        print("Streaming completed!")


if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'fraud_detection',
        'user': 'mlops_user',
        'password': 'mlops_password'
    }

    csv_path = '../../data/creditcard.csv'

    streamer = TransactionStreamer(csv_path, db_config)
    streamer.stream_transactions(batch_size=50, delay=1)
