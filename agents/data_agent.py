from common.schemas import AgentState

def fetch_data_features(state: AgentState) -> AgentState:
    """Stub for Data Agent"""
    state.data_features = {
        "account_age_days": state.request.account.account_age_days,
        "return_to_order_ratio": state.request.account.return_to_order_ratio,
        "prior_fraud_flags": state.request.account.prior_fraud_flags,
        "trust_tier": "low_trust" if state.request.account.account_age_days < 30 else "high_trust"
    }
    return state
