from common.schemas import AgentState

def check_image(state: AgentState) -> AgentState:
    """Stub for Image Check Agent"""
    if state.request.image_ref:
        state.image_flag = "consistency_ok"
    else:
        state.image_flag = "no_image"
    return state
