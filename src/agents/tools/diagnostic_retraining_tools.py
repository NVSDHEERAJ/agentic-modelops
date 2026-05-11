import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import settings
from src.logging.db import DatabaseManager
from src.monitoring.drift_detector import DriftDetector
from src.training.model_registry import ModelRegistry
from src.training.model_trainer import ModelTrainer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_version() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"candidate_{timestamp}"


def _load_labeled_production_data(
    db_manager: DatabaseManager,
    deployment_id: int,
    cutoff_timestamp: str,
) -> pd.DataFrame:
    query = """
    SELECT features_json, actual_label
    FROM prediction_logs
    WHERE deployment_id = ?
    AND actual_label IS NOT NULL
    AND timestamp <= ?;
    """

    with db_manager.get_connection() as conn:
        logs_df = pd.read_sql_query(
            query,
            conn,
            params=(deployment_id, cutoff_timestamp),
        )

    if logs_df.empty:
        return pd.DataFrame()

    features_df = pd.DataFrame(logs_df["features_json"].apply(json.loads).tolist())
    features_df[settings.TARGET_COL] = logs_df["actual_label"].astype(int)
    return features_df


def _build_candidate_dataset(
    db_manager: DatabaseManager,
    active_deployment: dict[str, Any],
    cutoff_timestamp: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    train_df = pd.read_csv(settings.TRAIN_PATH)
    validation_df = pd.read_csv(settings.VALIDATION_PATH)
    production_df = _load_labeled_production_data(
        db_manager=db_manager,
        deployment_id=int(active_deployment["id"]),
        cutoff_timestamp=cutoff_timestamp,
    )

    original_df = pd.concat([train_df, validation_df], ignore_index=True)
    combined_df = (
        pd.concat([original_df, production_df], ignore_index=True)
        if not production_df.empty
        else original_df
    )

    feature_columns = [
        column
        for column in combined_df.columns
        if column not in settings.DROP_COLS
    ]

    X = combined_df[feature_columns].fillna(-999)
    y = combined_df[settings.TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )

    dataset_summary = {
        "original_rows": int(len(original_df)),
        "labeled_production_rows": int(len(production_df)),
        "combined_rows": int(len(combined_df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "feature_count": int(len(feature_columns)),
        "training_data_cutoff_timestamp": cutoff_timestamp,
    }

    return X_train, y_train, X_val, y_val, dataset_summary


def _evaluate_model(model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
    y_probs = model.predict_proba(X_val)[:, 1]

    thresholds = [threshold / 100 for threshold in range(5, 95)]
    best_threshold = max(
        thresholds,
        key=lambda threshold: f1_score(
            y_val,
            (y_probs >= threshold).astype(int),
            zero_division=0,
        ),
    )
    y_pred = (y_probs >= best_threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_val, y_probs)),
        "pr_auc": float(average_precision_score(y_val, y_probs)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "best_threshold": float(best_threshold),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }


def _copy_preprocessing_artifact(
    active_deployment: dict[str, Any],
    candidate_dir: Path,
) -> Path:
    source_path = Path(
        active_deployment.get("preprocessing_path")
        or settings.DEFAULT_PREPROCESSING_PATH
    )
    destination_path = candidate_dir / source_path.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def _latest_candidate_summary_path() -> Path | None:
    if not settings.CANDIDATE_REPORT_DIR.exists():
        return None

    summaries = sorted(
        settings.CANDIDATE_REPORT_DIR.glob("candidate_*/candidate_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


@tool
def get_feature_drift_breakdown() -> dict[str, Any]:
    """
    Inspect which features are drifting the most for the active deployment.
    """
    db_manager = DatabaseManager()
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
    Retrain a challenger fraud detection model using the existing training pipeline.
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    active_deployment = db_manager.get_active_deployment()

    if active_deployment is None:
        return {
            "status": "failed",
            "candidate_model_trained": False,
            "active_deployment": None,
            "reason": "No active deployment is registered.",
        }

    cutoff_timestamp = _utc_now()
    candidate_version = _candidate_version()
    candidate_model_dir = settings.CANDIDATE_MODEL_DIR / candidate_version

    X_train, y_train, X_val, y_val, dataset_summary = _build_candidate_dataset(
        db_manager=db_manager,
        active_deployment=active_deployment,
        cutoff_timestamp=cutoff_timestamp,
    )

    trainer = ModelTrainer()
    candidate_model = trainer.train(X_train, y_train)
    candidate_metrics = _evaluate_model(candidate_model, X_val, y_val)

    preprocessing_path = _copy_preprocessing_artifact(
        active_deployment=active_deployment,
        candidate_dir=candidate_model_dir,
    )

    registry = ModelRegistry(
        model_version=candidate_version,
        model_dir=settings.CANDIDATE_MODEL_DIR,
        report_dir=settings.CANDIDATE_REPORT_DIR,
        preprocessing_path=preprocessing_path,
        db_manager=db_manager,
    )
    registry.save_model(candidate_model)
    registry.save_metadata(
        features=X_train.columns.tolist(),
        metrics=candidate_metrics,
        training_data_cutoff_timestamp=cutoff_timestamp,
    )
    registry.save_metrics(candidate_metrics)

    summary = {
        "status": "trained",
        "candidate_model_trained": True,
        "candidate_model_evaluated": True,
        "candidate_promoted": False,
        "candidate_version": candidate_version,
        "active_deployment": active_deployment,
        "training_data_cutoff_timestamp": cutoff_timestamp,
        "dataset_summary": dataset_summary,
        "candidate_metrics": candidate_metrics,
        "artifacts": {
            "model_path": str(registry.model_path),
            "metadata_path": str(registry.metadata_path),
            "preprocessing_path": str(preprocessing_path),
            "metrics_path": str(registry.metrics_path),
        },
        "note": (
            "Candidate model was trained and evaluated, but it was not promoted. "
            "Run evaluate_candidate_model to compare it with the active deployment."
        ),
    }

    summary_path = registry.versioned_report_dir / "candidate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return {
        **summary,
        "summary_path": str(summary_path),
    }


@tool
def evaluate_candidate_model() -> dict[str, Any]:
    """
    Evaluate challenger model and compare with current champion model.
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    active_deployment = db_manager.get_active_deployment()

    if active_deployment is None:
        return {
            "status": "failed",
            "candidate_model_evaluated": False,
            "candidate_promotable": False,
            "active_deployment": None,
            "reason": "No active deployment is registered.",
        }

    summary_path = _latest_candidate_summary_path()
    if summary_path is None:
        return {
            "status": "failed",
            "candidate_model_evaluated": False,
            "candidate_promotable": False,
            "active_deployment": active_deployment,
            "reason": "No candidate summary was found. Run retrain_candidate_model first.",
        }

    with open(summary_path, "r", encoding="utf-8") as file:
        candidate_summary = json.load(file)

    candidate_metrics = candidate_summary["candidate_metrics"]
    active_metrics_path = active_deployment.get("metrics_path")

    if not active_metrics_path:
        return {
            "status": "failed",
            "candidate_model_evaluated": True,
            "candidate_promotable": False,
            "active_deployment": active_deployment,
            "candidate_summary": candidate_summary,
            "reason": "Active deployment does not have a metrics_path.",
        }

    with open(active_metrics_path, "r", encoding="utf-8") as file:
        active_metrics = json.load(file)

    f1_delta = candidate_metrics["f1"] - active_metrics["f1"]
    recall_delta = candidate_metrics["recall"] - active_metrics["recall"]
    pr_auc_delta = candidate_metrics["pr_auc"] - active_metrics["pr_auc"]

    candidate_promotable = (
        f1_delta >= 0
        and recall_delta >= -0.02
        and pr_auc_delta >= -0.01
    )

    return {
        "status": "evaluated",
        "candidate_model_evaluated": True,
        "candidate_promotable": bool(candidate_promotable),
        "active_deployment": active_deployment,
        "candidate_version": candidate_summary["candidate_version"],
        "candidate_summary_path": str(summary_path),
        "active_metrics": active_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": {
            "f1": float(f1_delta),
            "recall": float(recall_delta),
            "pr_auc": float(pr_auc_delta),
        },
        "note": (
            "Candidate is eligible for promotion under the current heuristic."
            if candidate_promotable
            else "Candidate is not eligible for promotion under the current heuristic."
        ),
    }
