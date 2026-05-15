from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from src.logging.db import DatabaseManager
from src.retraining.candidate_evaluator import CandidateEvaluator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@tool
def inspect_active_deployment() -> dict[str, Any]:
    """
    Inspect the currently active production model deployment.
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    active_deployment = db_manager.get_active_deployment()

    if active_deployment is None:
        return {
            "status": "no_active_deployment",
            "active_deployment": None,
        }

    return {
        "status": "active_deployment_found",
        "active_deployment": active_deployment,
    }


@tool
def inspect_latest_candidate() -> dict[str, Any]:
    """
    Inspect the latest trained candidate model and its summary.
    """
    evaluator = CandidateEvaluator()
    loaded_candidate = evaluator.load_latest_candidate_summary()

    if loaded_candidate is None:
        return {
            "status": "no_candidate_found",
            "candidate_found": False,
            "candidate_summary": None,
        }

    candidate_summary, summary_path = loaded_candidate

    return {
        "status": "candidate_found",
        "candidate_found": True,
        "candidate_summary_path": str(summary_path),
        "candidate_summary": candidate_summary,
    }


@tool
def validate_candidate_for_promotion() -> dict[str, Any]:
    """
    Validate whether the latest candidate passes deterministic promotion gates.
    """
    evaluator = CandidateEvaluator()
    evaluation = evaluator.evaluate_latest_candidate()

    if evaluation.get("status") != "evaluated":
        return {
            **evaluation,
            "promotion_validated": False,
            "human_approval_required": True,
        }

    candidate_promotable = bool(evaluation.get("candidate_promotable", False))

    return {
        **evaluation,
        "promotion_validated": candidate_promotable,
        "human_approval_required": True,
        "note": (
            "Candidate passed deterministic gates. Human approval is required before promotion."
            if candidate_promotable
            else "Candidate failed deterministic gates and should not be promoted."
        ),
    }


@tool
def request_human_approval_for_promotion() -> dict[str, Any]:
    """
    Create a human approval request for promoting the latest candidate.

    This does not promote the model. It only packages the evidence for review.
    """
    evaluator = CandidateEvaluator()
    evaluation = evaluator.evaluate_latest_candidate()

    if evaluation.get("status") != "evaluated":
        return {
            "status": "approval_request_failed",
            "approval_requested": False,
            "human_approval_required": True,
            "reason": "Latest candidate could not be evaluated.",
            "evaluation": evaluation,
        }

    if not evaluation.get("candidate_promotable", False):
        return {
            "status": "approval_not_requested",
            "approval_requested": False,
            "human_approval_required": True,
            "reason": "Candidate failed deterministic promotion gates.",
            "evaluation": evaluation,
        }

    return {
        "status": "approval_requested",
        "approval_requested": True,
        "human_approval_required": True,
        "recommended_action": "approve_candidate_promotion",
        "candidate_version": evaluation["candidate_version"],
        "candidate_summary_path": evaluation["candidate_summary_path"],
        "active_deployment": evaluation["active_deployment"],
        "metric_deltas": evaluation["metric_deltas"],
        "promotion_criteria": evaluation["promotion_criteria"],
        "message": (
            "Candidate passed deterministic promotion gates. Human approval is required "
            "before registering it as the active deployment."
        ),
    }


@tool
def promote_latest_candidate_after_approval(approval_granted: bool) -> dict[str, Any]:
    """
    Promote the latest candidate only after explicit human approval.

    Args:
        approval_granted: Must be True to register the candidate as active.
    """
    if not approval_granted:
        return {
            "status": "promotion_blocked",
            "promoted": False,
            "human_approval_required": True,
            "reason": "Human approval was not granted.",
        }

    db_manager = DatabaseManager()
    db_manager.initialize_database()

    evaluator = CandidateEvaluator(db_manager=db_manager)
    evaluation = evaluator.evaluate_latest_candidate()

    if evaluation.get("status") != "evaluated":
        return {
            "status": "promotion_failed",
            "promoted": False,
            "reason": "Latest candidate could not be evaluated.",
            "evaluation": evaluation,
        }

    if not evaluation.get("candidate_promotable", False):
        return {
            "status": "promotion_blocked",
            "promoted": False,
            "reason": "Candidate failed deterministic promotion gates.",
            "evaluation": evaluation,
        }

    loaded_candidate = evaluator.load_latest_candidate_summary()

    if loaded_candidate is None:
        return {
            "status": "promotion_failed",
            "promoted": False,
            "reason": "No candidate summary found.",
        }

    candidate_summary, summary_path = loaded_candidate
    artifacts = candidate_summary["artifacts"]

    candidate_version = candidate_summary["candidate_version"]
    training_data_cutoff_timestamp = candidate_summary.get(
        "training_data_cutoff_timestamp"
    )

    deployment_id = db_manager.register_deployment(
        model_version=candidate_version,
        model_path=artifacts["model_path"],
        metadata_path=artifacts["metadata_path"],
        preprocessing_path=artifacts["preprocessing_path"],
        metrics_path=artifacts["metrics_path"],
        trained_at=training_data_cutoff_timestamp,
        deployed_at=_utc_now(),
        training_data_cutoff_timestamp=training_data_cutoff_timestamp,
        make_active=True,
    )

    return {
        "status": "promoted",
        "promoted": True,
        "deployment_id": deployment_id,
        "model_version": candidate_version,
        "candidate_summary_path": str(summary_path),
        "registered_paths": {
            "model_path": artifacts["model_path"],
            "metadata_path": artifacts["metadata_path"],
            "preprocessing_path": artifacts["preprocessing_path"],
            "metrics_path": artifacts["metrics_path"],
        },
        "evaluation": evaluation,
        "note": (
            "Candidate was registered as the active deployment. Restart or reload "
            "the serving process before sending production traffic."
        ),
    }


@tool
def rollback_to_previous_deployment(reason: str) -> dict[str, Any]:
    """
    Roll back to the most recent inactive deployment.

    Args:
        reason: Human-readable reason for rollback.
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    active_deployment = db_manager.get_active_deployment()

    query = """
    SELECT *
    FROM model_deployments
    WHERE is_active = 0
    ORDER BY deployed_at DESC
    LIMIT 1;
    """

    with db_manager.get_connection() as conn:
        conn.row_factory = None
        row = conn.execute(query).fetchone()

    if row is None:
        return {
            "status": "rollback_failed",
            "rolled_back": False,
            "reason": "No previous inactive deployment found.",
            "active_deployment": active_deployment,
        }

    previous_deployment_id = int(row[0])

    with db_manager.get_connection() as conn:
        conn.execute(
            """
            UPDATE model_deployments
            SET is_active = 0
            WHERE is_active = 1;
            """
        )
        conn.execute(
            """
            UPDATE model_deployments
            SET is_active = 1,
                status = 'active'
            WHERE id = ?;
            """,
            (previous_deployment_id,),
        )
        conn.commit()

    new_active_deployment = db_manager.get_active_deployment()

    return {
        "status": "rolled_back",
        "rolled_back": True,
        "rollback_reason": reason,
        "previous_active_deployment": active_deployment,
        "new_active_deployment": new_active_deployment,
        "note": (
            "Rollback updated the active deployment registry. Restart or reload "
            "the serving process before sending production traffic."
        ),
    }