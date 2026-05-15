import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import settings
from src.logging.db import DatabaseManager
from src.retraining.data_builder import RetrainingDataBuilder
from src.retraining.preprocessor import CandidatePreprocessor
from src.training.model_registry import ModelRegistry
from src.training.model_trainer import ModelTrainer


class CandidateRetrainingPipeline:
    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        self.db_manager = db_manager or DatabaseManager()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _candidate_version(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"candidate_{timestamp}"

    def _evaluate_model(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
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

    def run(self) -> dict[str, Any]:
        self.db_manager.initialize_database()

        active_deployment = self.db_manager.get_active_deployment()
        if active_deployment is None:
            return {
                "status": "failed",
                "candidate_model_trained": False,
                "reason": "No active deployment is registered.",
            }

        cutoff_timestamp = self._utc_now()
        candidate_version = self._candidate_version()

        candidate_data_dir = settings.DATA_DIR / "candidates" / candidate_version
        candidate_model_dir = settings.CANDIDATE_MODEL_DIR / candidate_version
        candidate_report_dir = settings.CANDIDATE_REPORT_DIR / candidate_version

        data_builder = RetrainingDataBuilder(db_manager=self.db_manager)
        combined_raw_df, data_summary = data_builder.build_combined_raw_dataset(
            active_deployment=active_deployment,
            cutoff_timestamp=cutoff_timestamp,
            output_dir=candidate_data_dir,
        )

        preprocessor = CandidatePreprocessor(
            data_output_dir=candidate_data_dir,
            artifact_output_dir=candidate_model_dir,
        )
        X_train, y_train, X_val, y_val, preprocessing_summary = (
            preprocessor.split_fit_transform(combined_raw_df)
        )

        trainer = ModelTrainer()
        candidate_model = trainer.train(X_train, y_train)

        candidate_metrics = self._evaluate_model(
            model=candidate_model,
            X_val=X_val,
            y_val=y_val,
        )

        registry = ModelRegistry(
            model_version=candidate_version,
            model_dir=settings.CANDIDATE_MODEL_DIR,
            report_dir=settings.CANDIDATE_REPORT_DIR,
            preprocessing_path=candidate_model_dir / "label_encoders.pkl",
            db_manager=self.db_manager,
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
            "data_summary": data_summary,
            "preprocessing_summary": preprocessing_summary,
            "candidate_metrics": candidate_metrics,
            "artifacts": {
                "candidate_data_dir": str(candidate_data_dir),
                "candidate_model_dir": str(candidate_model_dir),
                "candidate_report_dir": str(candidate_report_dir),
                "model_path": str(registry.model_path),
                "metadata_path": str(registry.metadata_path),
                "metrics_path": str(registry.metrics_path),
                "preprocessing_path": str(candidate_model_dir / "label_encoders.pkl"),
                "preprocessing_metadata_path": str(
                    candidate_model_dir / "preprocessing_metadata.json"
                ),
            },
        }

        candidate_report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = candidate_report_dir / "candidate_summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        return {
            **summary,
            "summary_path": str(summary_path),
        }