from common.schemas import AgentState

def evaluate_text_intent(state: AgentState) -> AgentState:
    """Stub for Text Agent"""
    reason = state.request.reason_text.lower()
    
    intent = {
        "implausible": False,
        "contradictory": False,
        "sarcastic": False,
        "rationale": "Reason appears normal."
    }
    
    if "broken" in reason and state.request.order.delivery_scan_confirmed:
        intent["contradictory"] = True
        intent["rationale"] = "Reason 'arrived broken' contradicts delivery_scan_confirmed = true"
        
    state.text_intent = intent
    return state
