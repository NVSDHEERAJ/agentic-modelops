from pprint import pprint

import requests

from scripts.generate_monitoring_report import main as generate_unified_report
from scripts.run_drift_detection import main as run_drift_detection
from scripts.run_monitoring import main as run_monitoring
from scripts.simulate_production_traffic import main as simulate_traffic
from src.agents.graph import build_modelops_graph
from src.agents.state import AgentState
from src.agents.tools.deployment_governance_tools import (
    promote_latest_candidate_after_approval,
    validate_candidate_for_promotion,
)
from langchain_core.messages import HumanMessage


def run_agent_graph() -> dict:
    graph = build_modelops_graph()

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "Run the full ModelOps loop for the production fraud model. "
                    "Inspect monitoring evidence, diagnose degradation if needed, "
                    "train/evaluate a candidate when justified, and request governance "
                    "approval if the candidate is promotable."
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

    return graph.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )


def reload_fastapi_model() -> None:
    response = requests.post("http://localhost:8000/reload-model", timeout=10)
    response.raise_for_status()
    print("\nFastAPI model reload result:")
    pprint(response.json())


def maybe_approve_latest_candidate() -> None:
    validation = validate_candidate_for_promotion.invoke({})

    print("\nCandidate validation:")
    pprint(validation)

    if not validation.get("promotion_validated", False):
        print("\nCandidate is not valid for promotion. No promotion requested.")
        return

    if not validation.get("human_approval_required", True):
        print("\nUnexpected: human approval not required. No automatic promotion.")
        return

    approval = input("\nCandidate is promotable. Type 'yes' to approve promotion: ")
    approval_granted = approval.strip().lower() == "yes"

    promotion_result = promote_latest_candidate_after_approval.invoke(
        {"approval_granted": approval_granted}
    )

    print("\nPromotion result:")
    pprint(promotion_result)

    if promotion_result.get("promoted", False):
        reload_fastapi_model()


def main() -> None:
    print("\n1. Simulating production traffic...")
    simulate_traffic()

    print("\n2. Running performance monitoring...")
    run_monitoring()

    print("\n3. Running drift detection...")
    run_drift_detection()

    print("\n4. Generating unified monitoring report...")
    generate_unified_report()

    print("\n5. Running agent graph...")
    graph_result = run_agent_graph()
    pprint(graph_result)

    print("\n6. Checking whether latest candidate needs approval...")
    maybe_approve_latest_candidate()


if __name__ == "__main__":
    main()