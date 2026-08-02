from common.schemas import AgentState

def retrieve_policy(state: AgentState) -> AgentState:
    """Stub for Policy RAG Agent"""
    category = state.request.order.category
    if category == "electronics":
        state.policy_retrieval = {
            "clause_id": "RET-10D",
            "text": "10-day window for electronics"
        }
    else:
        state.policy_retrieval = {
            "clause_id": "RET-30D",
            "text": "30-day window for general items"
        }
    return state
