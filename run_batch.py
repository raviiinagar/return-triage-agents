import json
from common.schemas import AgentState, ReturnRequest, Order, Account
from orchestrator.graph import build_graph

def main():
    # 1. Create a dummy request
    dummy_req = ReturnRequest(
        return_id="R12345",
        order=Order(
            order_id="O987",
            sku="ELEC-778",
            category="electronics",
            order_value=45000.0,
            order_date="2026-07-20",
            delivery_scan_confirmed=True,
            event_adjacent=False,
            days_since_delivery=8  # Within 10-day window
        ),
        account=Account(
            account_id="A55",
            account_age_days=14,
            total_orders=3,
            total_returns=2,
            return_to_order_ratio=0.67,
            prior_fraud_flags=0
        ),
        reason_text="item arrived broken",
        image_ref=None
    )

    initial_state = AgentState(request=dummy_req)

    # 2. Build and run graph
    app = build_graph()
    
    print("--- Running Adjudication Pipeline ---")
    result_state = app.invoke(initial_state.model_dump())
    
    # 3. Print Output JSON Contract
    final_response = result_state.get("final_response")
    if final_response:
        print("\n--- Final Adjudication Result ---")
        if hasattr(final_response, "model_dump"):
            print(json.dumps(final_response.model_dump(), indent=2))
        else:
            print(json.dumps(final_response, indent=2))
    else:
        print("Pipeline failed to produce a final response.")

if __name__ == "__main__":
    main()
