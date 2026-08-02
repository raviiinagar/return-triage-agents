from common.schemas import AgentState, AdjudicationResponse, PolicyCitation
import time

def fuse_decisions(state: AgentState) -> AgentState:
    """Stub for decision fusion and justification"""
    
    # Calculate fake fused confidence
    fused_confidence = state.risk_score or 0.0
    if state.text_intent and state.text_intent.get("contradictory"):
        fused_confidence += 0.3
        
    # Verdict logic
    verdict = "Escalate"
    if fused_confidence > 0.7:
        verdict = "Auto-Reject"
    elif fused_confidence < 0.3:
        verdict = "Auto-Approve"

    justification = [
        f"account_age_days = {state.data_features.get('account_age_days')}" if state.data_features else "account features missing",
    ]
    if state.text_intent and state.text_intent.get("contradictory"):
        justification.append(state.text_intent.get("rationale"))
        
    policy_cit = PolicyCitation(
        clause_id=state.policy_retrieval["clause_id"] if state.policy_retrieval else "N/A",
        text=state.policy_retrieval["text"] if state.policy_retrieval else "N/A"
    )

    state.final_response = AdjudicationResponse(
        return_id=state.request.return_id,
        verdict=verdict,
        risk_score=state.risk_score or 0.0,
        fused_fraud_confidence=min(fused_confidence, 1.0),
        policy_citation=policy_cit,
        justification_trail=justification,
        escalation_dossier="Stub dossier for reviewer" if verdict == "Escalate" else None,
        latency_ms=840
    )
    return state
