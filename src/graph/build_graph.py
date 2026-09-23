# Connects agents into a workflow
"""Builds the Northstar-FinGuard LangGraph pipeline: a linear chain of seven agents that takes
a raw draft communication and produces a routed, CCO-ready compliance review.

Usage:
    from src.graph.build_graph import build_graph
    graph = build_graph()
    result = graph.invoke({"source_text": "..."})
"""

from langgraph.graph import END, StateGraph

from src.agents.claim_extraction_agent import claim_extraction_agent
from src.agents.compliance_risk_agent import compliance_risk_agent
from src.agents.evidence_retrieval_agent import evidence_retrieval_agent
from src.agents.executive_summary_agent import executive_summary_agent
from src.agents.grounding_verification_agent import grounding_verification_agent
from src.agents.intake_agent import intake_agent
from src.agents.review_routing_agent import review_routing_agent
from src.graph.state import ComplianceState

# Builds the Northstar-FinGuard LangGraph pipeline: a linear chain of seven agents that takes
# a raw draft communication and produces a routed, CCO-ready compliance review.
def build_graph():
    # Create a StateGraph with the ComplianceState schema, which defines the structure of the shared state 
    # passed between agents in the pipeline.
    graph = StateGraph(ComplianceState)

    # Add nodes for each agent in the pipeline, specifying the agent's name and the function that will 
    # be executed at each node.
    graph.add_node("intake", intake_agent)
    graph.add_node("claim_extraction", claim_extraction_agent)
    graph.add_node("evidence_retrieval", evidence_retrieval_agent)
    graph.add_node("grounding_verification", grounding_verification_agent)
    graph.add_node("compliance_risk", compliance_risk_agent)
    graph.add_node("review_routing", review_routing_agent)
    graph.add_node("executive_summary", executive_summary_agent)

    # Define the starting point and edges between the nodes, specifying the order in which the agents will be executed.
    graph.set_entry_point("intake")
    graph.add_edge("intake", "claim_extraction")
    graph.add_edge("claim_extraction", "evidence_retrieval")
    graph.add_edge("evidence_retrieval", "grounding_verification")
    graph.add_edge("grounding_verification", "compliance_risk")
    graph.add_edge("compliance_risk", "review_routing")
    graph.add_edge("review_routing", "executive_summary")
    graph.add_edge("executive_summary", END)

    # Compile the graph, which prepares it for execution by validating the structure and ensuring that all nodes 
    # and edges are correctly defined.
    return graph.compile()

# Only if this script is run directly (not imported as a module), execute the following code.
if __name__ == "__main__":
    # Import the json module, which is used for formatting the output of the final state in a readable JSON format.
    import json

    sample = (
        "Northstar Wealth Partners is proud to say our growth portfolio beat the market "
        "every year for the last decade. Sign up today for free portfolio management, "
        "and see why we're the #1 rated advisor in the region."
    )
    # Build the LangGraph pipeline by calling the build_graph function, which returns a compiled graph 
    # ready for execution.
    app = build_graph()

    # Execute the graph with a sample input, which simulates the processing of a raw draft communication 
    # through the entire pipeline.
    final_state = app.invoke({"source_text": sample})

    # Print the final state of the pipeline after processing the sample input, formatted as a JSON string
    # with indentation for readability. The default=str argument ensures that any non-serializable objects
    # are converted to strings.
    print(json.dumps(final_state, indent=2, default=str))
