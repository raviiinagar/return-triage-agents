import sqlite3
import os

def init_db(db_path: str):
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_id TEXT PRIMARY KEY,
        account_age_days INTEGER,
        total_orders INTEGER,
        total_returns INTEGER,
        return_to_order_ratio REAL,
        prior_fraud_flags INTEGER,
        trust_tier TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        account_id TEXT,
        sku TEXT,
        category TEXT,
        order_value REAL,
        order_date TEXT,
        delivery_scan_confirmed BOOLEAN,
        event_adjacent BOOLEAN,
        FOREIGN KEY(account_id) REFERENCES accounts(account_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS return_requests (
        return_id TEXT PRIMARY KEY,
        order_id TEXT,
        reason_text TEXT,
        reason_tone TEXT,
        days_since_delivery INTEGER,
        image_ref TEXT,
        true_nature TEXT,
        fraud_archetype TEXT,
        governing_policy_clause_id TEXT,
        FOREIGN KEY(order_id) REFERENCES orders(order_id)
    )
    """)

    # Clear existing data for seed
    cursor.execute("DELETE FROM return_requests")
    cursor.execute("DELETE FROM orders")
    cursor.execute("DELETE FROM accounts")

    # Insert seed rows
    cursor.execute("""
    INSERT INTO accounts VALUES
    ('A55', 14, 3, 2, 0.67, 0, 'low_trust'),
    ('A99', 365, 50, 1, 0.02, 0, 'high_trust')
    """)

    cursor.execute("""
    INSERT INTO orders VALUES
    ('O987', 'A55', 'ELEC-778', 'electronics', 45000, '2026-07-20', 1, 0),
    ('O123', 'A99', 'TSHIRT-1', 'clothing', 1500, '2026-07-25', 1, 0)
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path} with seed data.")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "mock_orders.db")
    init_db(db_path)
