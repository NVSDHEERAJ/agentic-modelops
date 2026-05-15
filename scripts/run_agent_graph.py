from pprint import pprint

from langchain_core.messages import HumanMessage

from src.agents.graph import build_modelops_graph
from src.agents.state import AgentState


def main():
    graph = build_modelops_graph()

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "You are monitoring the production fraud detection model. "
                    "Use available tools to inspect model health, diagnose issues, "
                    "and decide the next action. If diagnosis is needed, investigate "
                    "drift, data quality, candidate retraining, and candidate evaluation."
                )
            )
        ],
        "observations": [],
        "tool_results": {},
        "agent_decisions": [],
        "iteration_count": 0,
        "max_iterations": 10,
        "errors": [],
    }

    result = graph.invoke(
        initial_state,
        config={
            "recursion_limit": 25,
        },
    )

    pprint(result)


if __name__ == "__main__":
    main()