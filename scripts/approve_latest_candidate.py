from pprint import pprint

from src.agents.tools.deployment_governance_tools import (
    promote_latest_candidate_after_approval,
    validate_candidate_for_promotion,
)


def main() -> None:
    validation = validate_candidate_for_promotion.invoke({})

    print("\nLatest candidate validation:")
    pprint(validation)

    if not validation.get("promotion_validated", False):
        print("\nCandidate is not valid for promotion. No action taken.")
        return

    candidate_version = validation.get("candidate_version")
    metric_deltas = validation.get("metric_deltas")

    print("\nCandidate passed deterministic promotion gates.")
    print(f"Candidate version: {candidate_version}")
    print("Metric deltas:")
    pprint(metric_deltas)

    approval = input("\nType 'yes' to approve promotion: ").strip().lower()

    result = promote_latest_candidate_after_approval.invoke(
        {
            "approval_granted": approval == "yes",
        }
    )

    print("\nPromotion result:")
    pprint(result)


if __name__ == "__main__":
    main()