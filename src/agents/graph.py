from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.state import AgentState
from src.agents.nodes.monitoring_agent import monitoring_agent, MONITORING_TOOLS
from src.agents.nodes.monitoring_decision import monitoring_decision_node
#from src.agents.nodes.diagnostic_retraining_agent import diagnostic_retraining_agent
#from src.agents.nodes.deployment_governance_agent import deployment_governance_agent

def route_after_monitoring_decision(state : AgentState) -> str:
    next_agent = state.get("next_agent", "end")

    if next_agent == "diagnostic_retraining_agent":
        return "diagnostic_retraining_agent"
    
    return END

def placeholder_diagnostic_retraining_agent(state : AgentState) -> AgentState:
    return {
        "current_agent" : "diagnostic_retraining_agent",
        "next_agent" : END,
        "final_summary" : (
            "Monitoring requested dignosis/retraining"
            "Diagnostic + Retraining Agent will be implemented next"
        )
    }

def build_modelops_graph():
    graph = StateGraph(AgentState)

    graph.add_node("monitoring_agent", monitoring_agent)
    graph.add_node("monitoring_tools", ToolNode(MONITORING_TOOLS))
    graph.add_node("monitoring_decision", monitoring_decision_node)
    graph.add_node(
        "diagnostic_retraining_agent",
        placeholder_diagnostic_retraining_agent
    )
    #graph.add_node("diagnostic_retraining_agent", diagnostic_retraining_agent)
    #graph.add_node("deployment_governance_agent", deployment_governance_agent)

    #graph.set_entry_point("monitoring_agent")

    graph.add_edge(START, "monitoring_agent")

    graph.add_conditional_edges(
        "monitoring_agent",
        tools_condition,
        {
            "tools" : "monitoring_tools",
            "__end__" : "monitoring_decision"
        }
    )

    graph.add_edge("monitoring_tools", "monitoring_decision")

    graph.add_conditional_edges(
        "monitoring_decision",
        route_after_monitoring_decision,
        {
            "diagnostic_retraining_agent" : "diagnostic_retraining_agent",
            END : END
        }
    )

    graph.add_edge("diagnostic_retraining_agent", END)
    #graph.add_edge("deployment_governance_agent", END)

    return graph.compile()