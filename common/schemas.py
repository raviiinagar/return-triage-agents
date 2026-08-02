from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Order(BaseModel):
    order_id: str
    sku: str
    category: str
    order_value: float
    order_date: str
    delivery_scan_confirmed: bool
    event_adjacent: bool
    days_since_delivery: int

class Account(BaseModel):
    account_id: str
    account_age_days: int
    total_orders: int
    total_returns: int
    return_to_order_ratio: float
    prior_fraud_flags: int

class ReturnRequest(BaseModel):
    return_id: str
    order: Order
    account: Account
    reason_text: str
    image_ref: Optional[str] = None

class PolicyCitation(BaseModel):
    clause_id: str
    text: str

class AdjudicationResponse(BaseModel):
    return_id: str
    verdict: str
    risk_score: float
    fused_fraud_confidence: float
    policy_citation: PolicyCitation
    justification_trail: List[str]
    escalation_dossier: Optional[str] = None
    latency_ms: int

class AgentState(BaseModel):
    request: ReturnRequest
    deterministic_result: Optional[Dict[str, Any]] = None
    data_features: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    text_intent: Optional[Dict[str, Any]] = None
    policy_retrieval: Optional[Dict[str, Any]] = None
    image_flag: Optional[str] = None
    final_response: Optional[AdjudicationResponse] = None
