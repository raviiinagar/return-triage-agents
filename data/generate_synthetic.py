import sqlite3
import os
import random
import uuid
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm_client import generate_completion

def generate_reason(archetype: str, true_nature: str, order_value: float, category: str, days_since_delivery: int) -> tuple[str, str]:
    """Uses LLM to generate a realistic return reason and determine its tone."""
    prompt = f"""
    You are generating synthetic e-commerce return request text for a dataset.
    Generate a short (1-2 sentences) realistic return reason from a customer.
    
    Context:
    - Product Category: {category}
    - Order Value: ₹{order_value}
    - Days since delivery: {days_since_delivery}
    - Fraud Archetype: {archetype} (Options: none, wardrobing, empty_box, serial_returner)
    - True Nature: {true_nature} (Legitimate, Fraud, Borderline)
    
    Output JSON exactly in this format:
    {{
        "reason_text": "the reason they wrote",
        "reason_tone": "normal, evasive, sarcastic, or contradictory"
    }}
    """
    
    res = generate_completion(prompt, model="google/gemini-2.5-flash")
    # Clean the response to parse json
    if res.startswith("```json"):
        res = res.split("```json")[1].split("```")[0].strip()
    elif res.startswith("```"):
        res = res.split("```")[1].strip()
        
    try:
        data = json.loads(res)
        return data.get("reason_text", "Item was not what I expected"), data.get("reason_tone", "normal")
    except Exception as e:
        print(f"Failed to parse LLM output: {res}")
        return "Item defective", "normal"


def generate_dataset(db_path: str, num_records: int = 10):
    """Generates synthetic rows and populates the DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    categories = ["electronics", "clothing", "grocery", "toys", "furniture"]
    archetypes = ["none", "wardrobing", "empty_box", "serial_returner"]
    natures = ["Legitimate", "Fraud", "Borderline"]
    
    for i in range(num_records):
        print(f"Generating record {i+1}/{num_records}...")
        
        account_id = f"A_{uuid.uuid4().hex[:6]}"
        order_id = f"O_{uuid.uuid4().hex[:6]}"
        return_id = f"R_{uuid.uuid4().hex[:6]}"
        
        nature = random.choice(natures)
        
        # Default logical values
        age_days = random.randint(30, 1000)
        tot_orders = random.randint(1, 100)
        tot_returns = random.randint(0, int(tot_orders * 0.2))
        fraud_flags = 0
        
        category = random.choice(categories)
        order_value = round(random.uniform(500, 50000), 2)
        days_since_delivery = random.randint(1, 40)
        scan = True
        event_adj = False
        
        archetype = "none"
        
        # Shape data based on nature
        if nature == "Fraud":
            archetype = random.choice(["wardrobing", "empty_box", "serial_returner"])
            if archetype == "wardrobing":
                category = "clothing"
                order_value = random.uniform(15000, 50000)
                event_adj = True
                days_since_delivery = random.randint(1, 3)
            elif archetype == "empty_box":
                scan = True
                age_days = random.randint(300, 1000) # Old account suddenly doing this
            elif archetype == "serial_returner":
                tot_returns = int(tot_orders * random.uniform(0.5, 0.9)) # >40% ratio
                
        # Calculate derived fields
        ratio = round(tot_returns / tot_orders, 2) if tot_orders > 0 else 0.0
        trust = "high_trust" if age_days > 90 and ratio < 0.2 and fraud_flags == 0 else "low_trust"
        
        reason, tone = generate_reason(archetype, nature, order_value, category, days_since_delivery)
        
        cursor.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (account_id, age_days, tot_orders, tot_returns, ratio, fraud_flags, trust))
            
        cursor.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (order_id, account_id, f"SKU-{random.randint(100, 999)}", category, order_value, "2026-08-01", scan, event_adj))
            
        cursor.execute("INSERT INTO return_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (return_id, order_id, reason, tone, days_since_delivery, None, nature, archetype, "MOCK-CLAUSE"))
            
    conn.commit()
    conn.close()
    print(f"Generated {num_records} records.")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "mock_orders.db")
    generate_dataset(db_path, num_records=10)
