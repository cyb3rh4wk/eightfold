from context import VectorDB
from groq_client import GroqClient
import json
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, TypedDict, Optional, List

class State(TypedDict):
    query: str
    history: List[str]
    extracted_info: Optional[dict]
    context: Optional[str]
    decision: Optional[str]
    response: Optional[str]
    error: Optional[str]


# Initialize groq client and vector db client
groq_client = GroqClient()
vector_db = VectorDB(collection_name="resolution_docs")

async def query_ingestion(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state.get("history", []).copy()
    history.append(state["query"])
    return {
        "history": history
    }

async def nlp_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        query = state["query"]
        history = state["history"]
        extracted_info = groq_client.extract_info(query, history[-3:])
        print('[DEBUG] LLM Extraction: ', extracted_info)
        state["extracted_info"] = extracted_info
        state["decision"] = extracted_info.get("decision")
    except Exception as e:
        print('[ERROR] NLP Error: ', str(e))
        state["error"] = f"NLP Processing Error: {str(e)}"
        state["decision"] = "escalate"
    return {k: v for k, v in state.items() if k != 'history'}

async def context_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        query = state["query"]
        relevant_docs = vector_db.search(query)
        state["context"] = "\n".join(relevant_docs) if relevant_docs else "No relevant context found."
        print('[DEBUG] Relevant Context: ', state["context"])
    except Exception as e:
        print('[ERROR] Context Retrieval Error: ', str(e))
        state["error"] = f"Context Retrieval Error: {str(e)}"

    return {k: v for k, v in state.items() if k != 'history'}

async def decision_making(state: Dict[str, Any]) -> Dict[str, Any]:
    print('[DEBUG] Decision Making State: ', state)
    if "error" in state or "extracted_info" not in state or state.get("decision") == "escalate":
        state["decision"] = "escalate"
    else:
        info = state["extracted_info"]
        if info.get("priority") == "High" or info.get("urgency") == "High":
            state["decision"] = "escalate"
        else:
            state["decision"] = "auto-respond"
    return {k: v for k, v in state.items() if k != 'history'}

async def response_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a response based on the decision."""
    if state.get("decision") == "auto-respond":
        query_context = state.get("context")
        extracted_info = state.get("extracted_info", {})
        
        final_response = groq_client.generate_final_response(
            query=state["query"],
            context=query_context,
            extracted_info=extracted_info
        )
        state["response"] = final_response.get("answer")
        state["decision"] = final_response.get("decision")
    
    print('[DEBUG] Final State: ', json.dumps(state, indent=2))

    return {k: v for k, v in state.items() if k != 'history'}

async def escalation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handles escalation to human agent."""
    state["response"] = "Your query has been escalated to a human agent. Please wait while we connect you to one of our customer delight representatives..."
    return {k: v for k, v in state.items() if k != 'history'}

# build the StateGraph
workflow = StateGraph(State)
workflow.add_node("query_ingestion", query_ingestion)
workflow.add_node("nlp_processing", nlp_processing)
workflow.add_node("context_retrieval", context_retrieval)
workflow.add_node("decision_making", decision_making)
workflow.add_node("response_generation", response_generation)
workflow.add_node("escalation", escalation)

# add edges
workflow.add_edge(START, "query_ingestion")
workflow.add_edge("query_ingestion", "nlp_processing")
workflow.add_edge("nlp_processing", "context_retrieval")
workflow.add_edge("context_retrieval", "decision_making")
workflow.add_edge("decision_making", "response_generation")
workflow.add_conditional_edges("response_generation", 
                               lambda state: state.get("decision") == "escalate", 
                               {True: "escalation", False: END})

# compile the graph
graph = workflow.compile()