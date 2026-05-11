import sqlite3
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from src.config import settings
from src.logging.db import DatabaseManager
from src.monitoring.monitoring_report import MonitoringReportGenerator


@tool
def generate_monitoring_report() -> dict[str, Any]:
    """
    Generate the latest unified monitoring report for the fraud detection model.
    Includes performance metrics, drift metrics, baseline_metrics, baseline status, and recommendation.
    """
    generator = MonitoringReportGenerator()
    return generator.generate_report()

@tool
def generate_traffic_stats() -> dict[str, Any]:
    """
    Get recent production traffic statistics for the active deployment.
    """
    db_path = settings.DB_PATH
    db_manager = DatabaseManager(db_path=db_path)
    active_deployment = db_manager.get_active_deployment()

    empty_response = {
        "status": "no_traffic",
        "source": str(db_path),
        "active_deployment": active_deployment,
        "total_predictions": 0,
        "anomaly_count": 0,
        "anomaly_rate": 0.0,
        "fraud_prediction_rate": 0.0,
        "average_fraud_probability": None,
        "min_fraud_probability": None,
        "max_fraud_probability": None,
        "labeled_predictions": 0,
        "unlabeled_predictions": 0,
        "model_version_counts": {},
        "latest_model_version": None,
        "deployment_id": active_deployment["id"] if active_deployment else None,
        "time_window": {
            "start": None,
            "end": None,
            "duration_hours": 0.0,
        },
    }

    if not db_path.exists():
        return empty_response

    if active_deployment is None:
        query = """
        SELECT
            timestamp,
            model_version,
            deployment_id,
            fraud_probability,
            prediction,
            actual_label
        FROM prediction_logs
        ORDER BY timestamp ASC;
        """
        params = ()
    else:
        query = """
        SELECT
            timestamp,
            model_version,
            deployment_id,
            fraud_probability,
            prediction,
            actual_label
        FROM prediction_logs
        WHERE deployment_id = ?
        ORDER BY timestamp ASC;
        """
        params = (active_deployment["id"],)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return empty_response
        raise

    if not rows:
        return empty_response

    total_predictions = len(rows)
    predictions = [int(row["prediction"]) for row in rows]
    fraud_probabilities = [float(row["fraud_probability"]) for row in rows]
    anomaly_count = sum(predictions)
    anomaly_rate = anomaly_count / total_predictions
    labeled_predictions = sum(row["actual_label"] is not None for row in rows)

    model_version_counts: dict[str, int] = {}
    for row in rows:
        model_version = str(row["model_version"])
        model_version_counts[model_version] = model_version_counts.get(model_version, 0) + 1

    start_timestamp = rows[0]["timestamp"]
    end_timestamp = rows[-1]["timestamp"]
    duration_hours = 0.0

    try:
        start_dt = datetime.fromisoformat(start_timestamp)
        end_dt = datetime.fromisoformat(end_timestamp)
        duration_hours = max((end_dt - start_dt).total_seconds() / 3600, 0.0)
    except ValueError:
        duration_hours = 0.0

    latest_deployment_id = rows[-1]["deployment_id"]

    return {
        "status": "traffic_observed",
        "source": str(db_path),
        "active_deployment": active_deployment,
        "total_predictions": total_predictions,
        "anomaly_count": int(anomaly_count),
        "anomaly_rate": float(anomaly_rate),
        "fraud_prediction_rate": float(anomaly_rate),
        "average_fraud_probability": float(sum(fraud_probabilities) / total_predictions),
        "min_fraud_probability": float(min(fraud_probabilities)),
        "max_fraud_probability": float(max(fraud_probabilities)),
        "labeled_predictions": int(labeled_predictions),
        "unlabeled_predictions": int(total_predictions - labeled_predictions),
        "model_version_counts": model_version_counts,
        "latest_model_version": rows[-1]["model_version"],
        "deployment_id": int(latest_deployment_id) if latest_deployment_id is not None else None,
        "time_window": {
            "start": start_timestamp,
            "end": end_timestamp,
            "duration_hours": float(duration_hours),
        },
    }

@tool
def check_last_retrain_date() -> dict[str, Any]:
    """
    Check when the current production model was last retrained.
    """
    db_manager = DatabaseManager()
    active_deployment = db_manager.get_active_deployment()

    if active_deployment is None:
        return {
            "status": "no_active_deployment",
            "last_retrain_date": None,
            "days_since_last_retrain": None,
            "retrain_pipeline_available": False,
            "active_deployment": None,
            "note": "No active model deployment is registered yet.",
        }

    trained_at = active_deployment.get("trained_at")
    days_since_last_retrain = None

    if trained_at:
        try:
            trained_at_dt = datetime.fromisoformat(trained_at)
            days_since_last_retrain = (
                datetime.now(trained_at_dt.tzinfo) - trained_at_dt
            ).days
        except ValueError:
            days_since_last_retrain = None

    model_version = active_deployment["model_version"]
    is_baseline = model_version == settings.DEFAULT_MODEL_VERSION

    return {
        "status": "baseline_never_retrained" if is_baseline else "retrained",
        "model_version": model_version,
        "deployment_id": active_deployment["id"],
        "last_retrain_date": None if is_baseline else trained_at,
        "trained_at": trained_at,
        "deployed_at": active_deployment.get("deployed_at"),
        "days_since_last_retrain": days_since_last_retrain,
        "retrain_pipeline_available": False,
        "active_deployment": active_deployment,
        "note": (
            "The active model is the original baseline and has not been retrained yet."
            if is_baseline
            else "The active model was created by a retraining workflow."
        ),
    }