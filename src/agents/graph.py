from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.nodes.deployment_governance_agent import (
    DEPLOYMENT_GOVERNANCE_TOOLS,
    deployment_governance_agent,
)
from src.agents.nodes.deployment_governance_decision import (
    deployment_governance_decision_node,
)
from src.agents.nodes.diagnostic_retraining_agent import (
    DIAGNOSTIC_RETRAINING_TOOLS,
    diagnostic_retraining_agent,
)
from src.agents.nodes.diagnostic_retraining_decision import (
    diagnostic_retraining_decision_node,
)
from src.agents.nodes.monitoring_agent import MONITORING_TOOLS, monitoring_agent
from src.agents.nodes.monitoring_decision import monitoring_decision_node
from src.agents.state import AgentState


def route_after_monitoring_decision(state: AgentState) -> str:
    next_agent = state.get("next_agent", "end")

    if next_agent == "diagnostic_retraining_agent":
        return "diagnostic_retraining_agent"

    return END


def route_after_diagnostic_retraining_decision(state: AgentState) -> str:
    next_agent = state.get("next_agent", "end")

    if next_agent == "deployment_governance_agent":
        return "deployment_governance_agent"

    return END

def build_modelops_graph():
    graph = StateGraph(AgentState)

    graph.add_node("monitoring_agent", monitoring_agent)
    graph.add_node("monitoring_tools", ToolNode(MONITORING_TOOLS))
    graph.add_node("monitoring_decision", monitoring_decision_node)

    graph.add_node("diagnostic_retraining_agent", diagnostic_retraining_agent)
    graph.add_node(
        "diagnostic_retraining_tools",
        ToolNode(DIAGNOSTIC_RETRAINING_TOOLS),
    )
    graph.add_node(
        "diagnostic_retraining_decision",
        diagnostic_retraining_decision_node,
    )

    graph.add_node("deployment_governance_agent", deployment_governance_agent)
    graph.add_node(
        "deployment_governance_tools",
        ToolNode(DEPLOYMENT_GOVERNANCE_TOOLS),
    )
    graph.add_node(
        "deployment_governance_decision",
        deployment_governance_decision_node,
    )

    graph.add_edge(START, "monitoring_agent")

    graph.add_conditional_edges(
        "monitoring_agent",
        tools_condition,
        {
            "tools": "monitoring_tools",
            "__end__": "monitoring_decision",
        },
    )

    graph.add_edge("monitoring_tools", "monitoring_decision")

    graph.add_conditional_edges(
        "monitoring_decision",
        route_after_monitoring_decision,
        {
            "diagnostic_retraining_agent": "diagnostic_retraining_agent",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "diagnostic_retraining_agent",
        tools_condition,
        {
            "tools": "diagnostic_retraining_tools",
            "__end__": "diagnostic_retraining_decision",
        },
    )

    graph.add_edge(
        "diagnostic_retraining_tools",
        "diagnostic_retraining_agent",
    )

    graph.add_conditional_edges(
        "diagnostic_retraining_decision",
        route_after_diagnostic_retraining_decision,
        {
            "deployment_governance_agent": "deployment_governance_agent",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "deployment_governance_agent",
        tools_condition,
        {
            "tools": "deployment_governance_tools",
            "__end__": "deployment_governance_decision",
        },
    )

    graph.add_edge(
        "deployment_governance_tools",
        "deployment_governance_agent",
    )

    graph.add_edge("deployment_governance_decision", END)

    return graph.compile()