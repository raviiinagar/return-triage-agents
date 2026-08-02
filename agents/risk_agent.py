from common.schemas import AgentState

def calculate_risk_score(state: AgentState) -> AgentState:
    """Stub for Risk Agent"""
    # Return fake risk score based on trust tier
    if state.data_features and state.data_features.get("trust_tier") == "low_trust":
        state.risk_score = 0.62
    else:
        state.risk_score = 0.15
    return state
