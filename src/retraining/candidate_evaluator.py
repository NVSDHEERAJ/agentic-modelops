import json
from pathlib import Path
from typing import Any

from src.config import settings
from src.logging.db import DatabaseManager


class CandidateEvaluator:
    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        self.db_manager = db_manager or DatabaseManager()

    def latest_candidate_summary_path(self) -> Path | None:
        if not settings.CANDIDATE_REPORT_DIR.exists():
            return None

        summaries = sorted(
            settings.CANDIDATE_REPORT_DIR.glob("candidate_*/candidate_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        return summaries[0] if summaries else None

    def load_latest_candidate_summary(self) -> tuple[dict[str, Any], Path] | None:
        summary_path = self.latest_candidate_summary_path()

        if summary_path is None:
            return None

        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)

        return summary, summary_path

    def load_active_metrics(
        self,
        active_deployment: dict[str, Any],
    ) -> dict[str, Any]:
        metrics_path = active_deployment.get("metrics_path")

        if not metrics_path:
            raise ValueError("Active deployment does not have a metrics_path")

        with open(metrics_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def compare_metrics(
        self,
        candidate_metrics: dict[str, Any],
        active_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        f1_delta = candidate_metrics["f1"] - active_metrics["f1"]
        recall_delta = candidate_metrics["recall"] - active_metrics["recall"]
        precision_delta = candidate_metrics["precision"] - active_metrics["precision"]
        pr_auc_delta = candidate_metrics["pr_auc"] - active_metrics["pr_auc"]
        roc_auc_delta = candidate_metrics["roc_auc"] - active_metrics["roc_auc"]

        candidate_promotable = (
            f1_delta >= 0
            and recall_delta >= -0.02
            and pr_auc_delta >= -0.01
        )

        return {
            "candidate_promotable": bool(candidate_promotable),
            "metric_deltas": {
                "f1": float(f1_delta),
                "recall": float(recall_delta),
                "precision": float(precision_delta),
                "pr_auc": float(pr_auc_delta),
                "roc_auc": float(roc_auc_delta),
            },
            "promotion_criteria": {
                "f1_delta_at_least": 0.0,
                "recall_delta_at_least": -0.02,
                "pr_auc_delta_at_least": -0.01,
            },
        }

    def evaluate_latest_candidate(self) -> dict[str, Any]:
        self.db_manager.initialize_database()
        active_deployment = self.db_manager.get_active_deployment()

        if active_deployment is None:
            return {
                "status": "failed",
                "candidate_model_evaluated": False,
                "candidate_promotable": False,
                "reason": "No active deployment is registered.",
            }

        loaded_candidate = self.load_latest_candidate_summary()
        if loaded_candidate is None:
            return {
                "status": "failed",
                "candidate_model_evaluated": False,
                "candidate_promotable": False,
                "active_deployment": active_deployment,
                "reason": "No candidate summary found. Run retrain_candidate_model first.",
            }

        candidate_summary, summary_path = loaded_candidate
        candidate_metrics = candidate_summary["candidate_metrics"]
        active_metrics = self.load_active_metrics(active_deployment)

        comparison = self.compare_metrics(
            candidate_metrics=candidate_metrics,
            active_metrics=active_metrics,
        )

        return {
            "status": "evaluated",
            "candidate_model_evaluated": True,
            "candidate_promotable": comparison["candidate_promotable"],
            "active_deployment": active_deployment,
            "candidate_version": candidate_summary["candidate_version"],
            "candidate_summary_path": str(summary_path),
            "active_metrics": active_metrics,
            "candidate_metrics": candidate_metrics,
            "metric_deltas": comparison["metric_deltas"],
            "promotion_criteria": comparison["promotion_criteria"],
            "note": (
                "Candidate is eligible for promotion under the current heuristic."
                if comparison["candidate_promotable"]
                else "Candidate is not eligible for promotion under the current heuristic."
            ),
        }
