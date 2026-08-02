from langgraph.graph import StateGraph, END
from common.schemas import AgentState
from orchestrator.deterministic import check_deterministic_rules
from agents.data_agent import fetch_data_features
from agents.risk_agent import calculate_risk_score
from agents.text_agent import evaluate_text_intent
from agents.policy_rag_agent import retrieve_policy
from agents.image_check import check_image
from orchestrator.fusion import fuse_decisions
from common.schemas import AdjudicationResponse, PolicyCitation
import time

def initial_validation(state: AgentState) -> AgentState:
    result = check_deterministic_rules(state.request.model_dump())
    state.deterministic_result = result
    
    # If hard-reject, we bypass agents and go straight to fusion (or early exit)
    # For Phase 1 stub, we'll let it flow through or handle it in fusion.
    if not result.get("pass"):
        # We can create a final response right here and bypass
        state.final_response = AdjudicationResponse(
            return_id=state.request.return_id,
            verdict="Auto-Reject",
            risk_score=1.0,
            fused_fraud_confidence=1.0,
            policy_citation=PolicyCitation(clause_id="N/A", text="N/A"),
            justification_trail=[result["reason"]],
            latency_ms=10
        )
    return state

def should_continue(state: AgentState):
    if state.final_response:
        return END
    return "parallel_agents"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("initial_validation", initial_validation)
    
    # Parallel agents
    workflow.add_node("data_agent", fetch_data_features)
    workflow.add_node("risk_agent", calculate_risk_score)
    workflow.add_node("text_agent", evaluate_text_intent)
    workflow.add_node("policy_agent", retrieve_policy)
    workflow.add_node("image_agent", check_image)
    
    # Fusion
    workflow.add_node("fusion", fuse_decisions)
    
    workflow.set_entry_point("initial_validation")
    
    # After initial validation, if passed, run agents in parallel
    # LangGraph doesn't have a built-in parallel node construct natively without creating a parallel execution branch,
    # but we can simulate it by connecting initial -> data -> risk -> text -> policy -> image -> fusion linearly for the stub,
    # or just run them sequentially for the stub since they don't depend on each other (except risk depends on data).
    # Correct DAG order: data -> risk. text, policy, image can be anywhere before fusion.
    
    workflow.add_conditional_edges("initial_validation", should_continue, {
        END: END,
        "parallel_agents": "data_agent"
    })
    
    workflow.add_edge("data_agent", "risk_agent")
    workflow.add_edge("risk_agent", "text_agent")
    workflow.add_edge("text_agent", "policy_agent")
    workflow.add_edge("policy_agent", "image_agent")
    workflow.add_edge("image_agent", "fusion")
    
    workflow.add_edge("fusion", END)
    
    return workflow.compile()
