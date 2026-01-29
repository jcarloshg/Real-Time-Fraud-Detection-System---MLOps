"""
Transaction Stream Simulator for Real-Time Fraud Detection System.

This module simulates streaming transaction data from a CSV file into a PostgreSQL database,
mimicking real-time transaction processing for fraud detection ML pipelines.
"""

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Optional

import pandas as pd
import psycopg
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransactionStreamer:
    """
    Simulates streaming transaction data into PostgreSQL database.

    This class reads transaction data from a CSV file and inserts it into a PostgreSQL
    database in batches, simulating real-time data streaming with configurable delays.

    Attributes:
        csv_path: Path to the CSV file containing transaction data.
        db_config: Database connection configuration dictionary.
        df: Pandas DataFrame containing the transaction data.
        current_index: Current position in the DataFrame for streaming.
    """

    def __init__(self, csv_path: str, db_config: Dict[str, str]) -> None:
        """
        Initialize the TransactionStreamer.

        Args:
            csv_path: Path to the CSV file with transaction data.
            db_config: Dictionary containing database connection parameters
                      (host, port, database, user, password).

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the CSV file is empty or invalid.
        """
        self.csv_path = Path(csv_path)
        self.db_config = db_config
        self.current_index = 0

        # Validate and load CSV
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            self.df = pd.read_csv(self.csv_path)
            if self.df.empty:
                raise ValueError(f"CSV file is empty: {csv_path}")
            logger.info(f"Loaded {len(self.df)} transactions from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator[psycopg.Connection, None, None]:
        """
        Context manager for database connections.

        Ensures proper connection cleanup even if exceptions occur.

        Yields:
            psycopg.Connection: Active database connection.

        Raises:
            psycopg.Error: If connection fails.
        """
        conn = None
        try:
            conn = psycopg.connect(**self.db_config)
            yield conn
        except psycopg.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def create_table(self) -> None:
        """
        Create the transactions table if it doesn't exist.

        The table includes an auto-incrementing ID, timestamp, all transaction features
        (time, v1-v28, amount) and the class label for fraud detection.

        Raises:
            psycopg.Error: If table creation fails.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            "Time" FLOAT,
            "V1" FLOAT, "V2" FLOAT, "V3" FLOAT, "V4" FLOAT, "V5" FLOAT,
            "V6" FLOAT, "V7" FLOAT, "V8" FLOAT, "V9" FLOAT, "V10" FLOAT,
            "V11" FLOAT, "V12" FLOAT, "V13" FLOAT, "V14" FLOAT, "V15" FLOAT,
            "V16" FLOAT, "V17" FLOAT, "V18" FLOAT, "V19" FLOAT, "V20" FLOAT,
            "V21" FLOAT, "V22" FLOAT, "V23" FLOAT, "V24" FLOAT, "V25" FLOAT,
            "V26" FLOAT, "V27" FLOAT, "V28" FLOAT,
            "Amount" FLOAT,
            "Class" INTEGER
        );
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_query)
                    conn.commit()
            logger.info("Table 'transactions' created successfully")
        except psycopg.Error as e:
            logger.error(f"Failed to create table: {e}")
            raise

    def stream_transactions(
        self,
        batch_size: int = 10,
        delay: float = 2.0
    ) -> None:
        """
        Stream transactions to the database in batches.

        Simulates real-time data streaming by inserting batches of transactions
        with a configurable delay between batches.

        Args:
            batch_size: Number of transactions to insert per batch. Defaults to 10.
            delay: Delay in seconds between batches. Defaults to 2.0.

        Raises:
            psycopg.Error: If database operations fail.
            KeyboardInterrupt: If streaming is interrupted by user.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if delay < 0:
            raise ValueError("delay must be non-negative")

        # Ensure table exists
        self.create_table()

        logger.info(
            f"Starting to stream {len(self.df)} transactions "
            f"(batch_size={batch_size}, delay={delay}s)"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    while self.current_index < len(self.df):
                        # Get batch efficiently using iloc
                        batch_end = min(
                            self.current_index + batch_size,
                            len(self.df)
                        )
                        batch = self.df.iloc[self.current_index:batch_end]

                        # Convert to tuples efficiently (avoid iterrows)
                        values = [tuple(row) for row in batch.to_numpy()]

                        # Prepare and execute insert using psycopg3's executemany
                        # Quote column names to preserve case sensitivity
                        columns = ','.join([f'"{col}"' for col in self.df.columns])
                        placeholders = ','.join(['%s'] * len(self.df.columns))
                        insert_query = f"INSERT INTO transactions ({columns}) VALUES ({placeholders})"

                        cur.executemany(insert_query, values)
                        conn.commit()

                        self.current_index = batch_end
                        progress = (self.current_index / len(self.df)) * 100
                        logger.info(
                            f"Inserted batch of {len(batch)} transactions. "
                            f"Progress: {self.current_index}/{len(self.df)} ({progress:.1f}%)"
                        )

                        # Simulate streaming delay
                        if self.current_index < len(self.df):
                            time.sleep(delay)

            logger.info("Streaming completed successfully!")

        except KeyboardInterrupt:
            logger.warning(
                f"Streaming interrupted by user at {self.current_index}/{len(self.df)}"
            )
            raise
        except psycopg.Error as e:
            logger.error(f"Database error during streaming: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}")
            raise


def get_db_config_from_env() -> Dict[str, str]:
    """
    Load database configuration from environment variables.

    Expected environment variables:
        - DB_HOST: Database host (default: localhost)
        - DB_PORT: Database port (default: 5432)
        - DB_NAME: Database name (default: fraud_detection)
        - DB_USER: Database user (default: mlops_user)
        - DB_PASSWORD: Database password (default: mlops_password)

    Returns:
        Dictionary containing database configuration.
    """
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'dbname': os.getenv('DB_NAME', 'fraud_detection'),
        'user': os.getenv('DB_USER', 'mlops_user'),
        'password': os.getenv('DB_PASSWORD', 'mlops_password')
    }


def main() -> None:
    """Main entry point for the transaction streamer."""
    # Load environment variables from .env file
    load_dotenv()

    # Load configuration
    db_config = get_db_config_from_env()

    # Get CSV path from environment or use default
    csv_path = os.getenv('CSV_PATH', '../../data/creditcard.csv')

    # Get streaming parameters from environment or use defaults
    batch_size = int(os.getenv('BATCH_SIZE', '50'))
    delay = float(os.getenv('STREAM_DELAY', '1.0'))

    try:
        # Initialize and run streamer
        streamer = TransactionStreamer(csv_path, db_config)
        streamer.stream_transactions(batch_size=batch_size, delay=delay)
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    except psycopg.Error as e:
        logger.error(f"Database error: {e}")
        exit(1)
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user")
        exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
