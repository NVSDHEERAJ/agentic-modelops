from typing import Any

from langchain_core.tools import tool

from src.logging.db import DatabaseManager
from src.monitoring.drift_detector import DriftDetector
from src.retraining.candidate_evaluator import CandidateEvaluator
from src.retraining.candidate_pipeline import CandidateRetrainingPipeline


@tool
def get_feature_drift_breakdown() -> dict[str, Any]:
    """
    Inspect which features are drifting the most for the active deployment.
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    active_deployment = db_manager.get_active_deployment()
    detector = DriftDetector(db_manager=db_manager)
    drift_report = detector.run()

    return {
        "active_deployment": active_deployment,
        "drift_report": drift_report,
    }


@tool
def run_data_quality_checks() -> dict[str, Any]:
    """
    Check whether the issue may be caused by missing values, schema mismatch, or bad input data.
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    active_deployment = db_manager.get_active_deployment()
    detector = DriftDetector(db_manager=db_manager)
    production_df = detector.load_production_data()
    model_features = detector.load_model_features()

    missing_features = [
        feature for feature in model_features if feature not in production_df.columns
    ]
    schema_valid = len(missing_features) == 0

    if schema_valid:
        missing_value_issue = production_df[model_features].isnull().any().any()
    else:
        missing_value_issue = True

    duplicate_rows_detected = production_df.duplicated().any()

    data_quality_status = (
        "passed"
        if schema_valid and not missing_value_issue and not duplicate_rows_detected
        else "failed"
    )

    return {
        "active_deployment": active_deployment,
        "schema_valid": bool(schema_valid),
        "missing_features": missing_features,
        "missing_value_issue": bool(missing_value_issue),
        "duplicate_rows_detected": bool(duplicate_rows_detected),
        "rows_checked": int(len(production_df)),
        "data_quality_status": data_quality_status,
    }


@tool
def retrain_candidate_model() -> dict[str, Any]:
    """
    Retrain a challenger fraud detection model using baseline raw data and labeled production logs.
    """
    pipeline = CandidateRetrainingPipeline()
    return pipeline.run()


@tool
def evaluate_candidate_model() -> dict[str, Any]:
    """
    Evaluate the latest challenger model and compare it with the active champion deployment.
    """
    evaluator = CandidateEvaluator()
    return evaluator.evaluate_latest_candidate()