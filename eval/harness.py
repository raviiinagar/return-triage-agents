import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schemas import AgentState, ReturnRequest, Order, Account
from orchestrator.graph import build_graph

# Hand-authored adversarial cases designed to test our stubbed pipeline logic.
ADVERSARIAL_CASES = [
    {
        "test_id": "TC-01",
        "description": "High trust account, normal reason -> Auto-Approve",
        "request": ReturnRequest(
            return_id="R-TC01",
            order=Order(order_id="O1", sku="S1", category="electronics", order_value=1000, order_date="2026-08-01", delivery_scan_confirmed=True, event_adjacent=False, days_since_delivery=2),
            account=Account(account_id="A1", account_age_days=365, total_orders=50, total_returns=1, return_to_order_ratio=0.02, prior_fraud_flags=0),
            reason_text="Item didn't fit properly.",
            image_ref=None
        ),
        "expected_verdict": "Auto-Approve"
    },
    {
        "test_id": "TC-02",
        "description": "Low trust account, normal reason -> Escalate",
        "request": ReturnRequest(
            return_id="R-TC02",
            order=Order(order_id="O2", sku="S2", category="clothing", order_value=20000, order_date="2026-08-01", delivery_scan_confirmed=True, event_adjacent=True, days_since_delivery=5),
            account=Account(account_id="A2", account_age_days=15, total_orders=2, total_returns=1, return_to_order_ratio=0.5, prior_fraud_flags=0),
            reason_text="Changed my mind.",
            image_ref=None
        ),
        "expected_verdict": "Escalate"
    },
    {
        "test_id": "TC-03",
        "description": "Low trust account, contradictory reason -> Auto-Reject",
        "request": ReturnRequest(
            return_id="R-TC03",
            order=Order(order_id="O3", sku="S3", category="electronics", order_value=50000, order_date="2026-08-01", delivery_scan_confirmed=True, event_adjacent=False, days_since_delivery=1),
            account=Account(account_id="A3", account_age_days=10, total_orders=1, total_returns=0, return_to_order_ratio=0.0, prior_fraud_flags=0),
            reason_text="The item arrived completely broken in pieces.",
            image_ref=None
        ),
        "expected_verdict": "Auto-Reject"
    },
    {
        "test_id": "TC-04",
        "description": "Deterministic Reject (Out of policy 15 days for electronics) -> Auto-Reject",
        "request": ReturnRequest(
            return_id="R-TC04",
            order=Order(order_id="O4", sku="S4", category="electronics", order_value=1500, order_date="2026-07-01", delivery_scan_confirmed=True, event_adjacent=False, days_since_delivery=15),
            account=Account(account_id="A4", account_age_days=365, total_orders=50, total_returns=1, return_to_order_ratio=0.02, prior_fraud_flags=0),
            reason_text="Doesn't work anymore.",
            image_ref=None
        ),
        "expected_verdict": "Auto-Reject"
    },
    {
        "test_id": "TC-05",
        "description": "High trust account, contradictory reason -> Escalate",
        "request": ReturnRequest(
            return_id="R-TC05",
            order=Order(order_id="O5", sku="S5", category="electronics", order_value=30000, order_date="2026-08-01", delivery_scan_confirmed=True, event_adjacent=False, days_since_delivery=2),
            account=Account(account_id="A5", account_age_days=500, total_orders=100, total_returns=2, return_to_order_ratio=0.02, prior_fraud_flags=0),
            reason_text="Screen arrived totally broken.",
            image_ref=None
        ),
        "expected_verdict": "Escalate"
    }
]

def run_eval():
    app = build_graph()
    results = []
    
    print(f"--- Starting Eval Harness with {len(ADVERSARIAL_CASES)} cases ---")
    
    for case in ADVERSARIAL_CASES:
        print(f"\nRunning {case['test_id']}: {case['description']}")
        
        initial_state = AgentState(request=case["request"])
        result_state = app.invoke(initial_state.model_dump())
        
        final_response = result_state.get("final_response")
        
        if final_response:
            # Handle Pydantic model vs dict
            if hasattr(final_response, "model_dump"):
                verdict = final_response.verdict
                fraud_conf = final_response.fused_fraud_confidence
            else:
                verdict = final_response.get("verdict")
                fraud_conf = final_response.get("fused_fraud_confidence")
                
            passed = (verdict == case["expected_verdict"])
            print(f"  Expected: {case['expected_verdict']} | Actual: {verdict} | Confidence: {fraud_conf:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            
            results.append(passed)
        else:
            print("  FAIL: Pipeline produced no final response.")
            results.append(False)
            
    accuracy = (sum(results) / len(results)) * 100
    print(f"\n--- Eval Complete: {accuracy:.1f}% Accuracy ---")

if __name__ == "__main__":
    run_eval()
