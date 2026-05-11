from pprint import pprint
from langchain_core.messages import HumanMessage

from src.agents.graph import build_modelops_graph

def main():
    graph = build_modelops_graph()

    result = graph.invoke({
        "messages" : [
            HumanMessage(
                content = (
                    "You are monitoring the production fraud detection model. "
                    "Use tools to inspect model health and decide the next action."
                )
            )
        ],
        "observations" : [],
        "tool_results" : {},
        "agent_decisions" : [],
        "iteration_count" : 0,
        "max_iterations" : 5,
        "errors" : []
    })

    pprint(result)

if __name__ == "__main__":
    main()