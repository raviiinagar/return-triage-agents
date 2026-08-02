# Return Triage Multi-Agent System

This repository contains the autonomous multi-agent triage system for Flipkart GRiD 8.0.
This document outlines the frozen decisions from Phase 0.

## 1. SQLite Schema (Mock Orders DB)
```sql
accounts(
  account_id PK, account_age_days, total_orders, total_returns,
  return_to_order_ratio, prior_fraud_flags, trust_tier
)
orders(
  order_id PK, account_id FK, sku, category, order_value,
  order_date, delivery_scan_confirmed BOOL, event_adjacent BOOL
)
return_requests(
  return_id PK, order_id FK, reason_text, reason_tone,
  days_since_delivery, image_ref NULLABLE,
  -- scoring fields:
  true_nature ENUM(Legitimate,Fraud,Borderline),
  fraud_archetype ENUM(none,wardrobing,empty_box,serial_returner),
  governing_policy_clause_id NULLABLE
)
```

## 2. Shared-State Object (LangGraph)
The agents communicate by updating a shared state dictionary containing:
- `request`: The raw input request.
- `deterministic_result`: Pre-check result (pass/hard-reject).
- `data_features`: Behavioral feature vector for the risk model.
- `risk_score`: Calibrated fraud probability [0-1] from Risk Agent.
- `text_intent`: Flags (implausible, contradictory, sarcastic) + rationale from Text Agent.
- `policy_retrieval`: Retrieved governing clause + `clause_id` from RAG Agent.
- `image_flag`: Consistency flag from optional Image Check.
- `final_response`: Final fused decision including `verdict` with `justification_trail` and `escalation_dossier`.

## 3. Output JSON Contract
```json
{
  "return_id": "R12345",
  "verdict": "Escalate",
  "risk_score": 0.62,
  "fused_fraud_confidence": 0.58,
  "policy_citation": {"clause_id": "RET-10D", "text": "10-day window for electronics"},
  "justification_trail": [
    "account_age_days = 14 (below trust threshold of 90)",
    "return_to_order_ratio = 0.62 (exceeds 0.40 serial-returner flag)",
    "reason 'arrived broken' contradicts delivery_scan_confirmed = true",
    "policy RET-10D satisfied (day 6 of 10)"
  ],
  "escalation_dossier": "…summary for human reviewer…",
  "latency_ms": 840
}
```

## 4. Verdict Mapping & Ground Truth
**Ground Truth (nature):**
- Legitimate
- Fraud
- Borderline

**System Prediction (verdict):**
- Auto-Approve
- Auto-Reject
- Escalate

**Mapping rules (Thresholds on `fused_fraud_confidence`):**
- **High risk + policy violation + contradictory reason → Auto-Reject**
- **Low risk + clean reason + policy-compliant → Auto-Approve**
- **Conflicting signals, or confidence in the "grey band", or policy edge case → Escalate**
- **Escalate is NOT a wrong decision on a legitimate claim (it acts as an abstain).**

## 5. Compute / Demo Machine
- **Primary:** Google Cloud Run (Pattern B: Streamlit service importing orchestrator module)
- **Local:** FastAPI running locally + Streamlit for live demo.
