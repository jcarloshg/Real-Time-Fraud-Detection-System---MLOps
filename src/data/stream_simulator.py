import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import time
import random
from datetime import datetime
from tqdm import tqdm
import argparse

class TransactionStreamer:
    """Simulates streaming transactions from the Kaggle dataset"""
    
    def __init__(self, csv_path, db_config):
        """
        Initialize the streamer
        
        Args:
            csv_path: Path to the creditcard.csv file
            db_config: Dictionary with database connection parameters
        """
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.db_config = db_config
        self.current_index = 0
        
        print(f"Dataset loaded: {len(self.df)} transactions")
        print(f"Fraud transactions: {self.df['Class'].sum()} ({self.df['Class'].mean()*100:.2f}%)")
        
    def connect_db(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def test_connection(self):
        """Test database connection"""
        try:
            conn = self.connect_db()
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"✅ Database connected: {version[0]}")
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def stream_transactions(self, batch_size=100, delay=1, max_transactions=None, shuffle=False):
        """
        Simulate streaming by inserting batches of transactions
        
        Args:
            batch_size: Number of transactions per batch
            delay: Seconds to wait between batches
            max_transactions: Maximum number of transactions to stream (None = all)
            shuffle: Whether to shuffle the data before streaming
        """
        
        # Shuffle if requested
        df_stream = self.df.copy()
        if shuffle:
            df_stream = df_stream.sample(frac=1, random_state=42).reset_index(drop=True)
            print("📊 Data shuffled for streaming")
        
        # Limit transactions if specified
        if max_transactions:
            df_stream = df_stream.head(max_transactions)
            print(f"📊 Limited to {max_transactions} transactions")
        
        conn = self.connect_db()
        cur = conn.cursor()
        
        total = len(df_stream)
        print(f"\n🚀 Starting to stream {total} transactions...")
        print(f"   Batch size: {batch_size}")
        print(f"   Delay: {delay}s between batches\n")
        
        # Progress bar
        pbar = tqdm(total=total, desc="Streaming", unit="tx")
        
        try:
            while self.current_index < total:
                # Get batch
                batch_end = min(self.current_index + batch_size, total)
                batch = df_stream.iloc[self.current_index:batch_end]
                
                # Prepare data for insertion
                values = []
                for idx, row in batch.iterrows():
                    # Generate unique transaction ID
                    tx_id = f"tx_{int(time.time()*1000)}_{idx}"
                    
                    # Build value tuple
                    value = (tx_id,) + tuple(row.values)
                    values.append(value)
                
                # Insert batch
                columns = "transaction_id, " + ', '.join(df_stream.columns)
                insert_query = f"""
                    INSERT INTO transactions ({columns})
                    VALUES %s
                    ON CONFLICT (transaction_id) DO NOTHING
                """
                
                execute_values(cur, insert_query, values)
                conn.commit()
                
                # Update progress
                batch_fraud = batch['Class'].sum()
                self.current_index = batch_end
                pbar.update(len(batch))
                pbar.set_postfix({
                    'fraud': f"{batch_fraud}/{len(batch)}",
                    'total': f"{self.current_index}/{total}"
                })
                
                # Simulate streaming delay
                if self.current_index < total:
                    time.sleep(delay)
            
            pbar.close()
            print(f"\n✅ Streaming completed!")
            print(f"   Total transactions inserted: {total}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Streaming interrupted by user")
            print(f"   Transactions inserted so far: {self.current_index}")
        except Exception as e:
            print(f"\n❌ Error during streaming: {e}")
        finally:
            cur.close()
            conn.close()

def main():
    """Main function to run the streamer"""
    
    parser = argparse.ArgumentParser(description='Stream fraud detection transactions to database')
    parser.add_argument('--csv', type=str, default='../../data/raw/creditcard.csv',
                       help='Path to creditcard.csv file')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of transactions per batch')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Seconds to wait between batches')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of transactions to stream')
    parser.add_argument('--shuffle', action='store_true',
                       help='Shuffle data before streaming')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'fraud_detection',
        'user': 'mlops_user',
        'password': 'mlops_password'
    }
    
    # Create streamer
    streamer = TransactionStreamer(args.csv, db_config)
    
    # Test connection
    if not streamer.test_connection():
        print("Please make sure PostgreSQL is running: docker-compose up -d postgres")
        return
    
    # Start streaming
    streamer.stream_transactions(
        batch_size=args.batch_size,
        delay=args.delay,
        max_transactions=args.max,
        shuffle=args.shuffle
    )

if __name__ == "__main__":
    main()