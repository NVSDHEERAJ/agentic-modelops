import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import settings
from src.logging.db import DatabaseManager


class ModelRegistry:
    def __init__(
        self,
        model_version: str = settings.DEFAULT_MODEL_VERSION,
        model_dir: Path = settings.MODEL_DIR,
        report_dir: Path = settings.REPORT_DIR,
        preprocessing_path: Path | None = None,
        db_manager: DatabaseManager | None = None,
    ):
        self.model_version = model_version
        self.model_dir = model_dir
        self.report_dir = report_dir
        self.preprocessing_path = preprocessing_path or settings.DEFAULT_PREPROCESSING_PATH
        self.db_manager = db_manager or DatabaseManager()

        self.versioned_model_dir = self.model_dir / self.model_version
        self.versioned_report_dir = self.report_dir / self.model_version

        self.model_path = self.versioned_model_dir / "model.pkl"
        self.metadata_path = self.versioned_model_dir / "metadata.json"
        self.metrics_path = self.versioned_report_dir / "metrics.json"

        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.training_data_cutoff_timestamp: str | None = None

    def save_model(self, model: Any) -> None:
        self.versioned_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

    def save_metadata(
        self,
        features: list[str],
        metrics: dict[str, Any],
        training_data_cutoff_timestamp: str | None = None,
    ) -> None:
        self.training_data_cutoff_timestamp = training_data_cutoff_timestamp

        metadata = {
            "model_version": self.model_version,
            "model_type": "LighGBMClassifier",
            "features": features,
            "threshold": metrics.get("best_threshold"),
            "metrics": metrics,
            "trained_at": self.trained_at,
            "training_data_cutoff_timestamp": training_data_cutoff_timestamp,
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "preprocessing_path": str(self.preprocessing_path),
            "metrics_path": str(self.metrics_path),
        }

        self.versioned_model_dir.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        self.versioned_report_dir.mkdir(parents=True, exist_ok=True)

        with open(self.metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

    def register_as_active_deployment(self) -> int:
        self.db_manager.initialize_database()

        return self.db_manager.register_deployment(
            model_version=self.model_version,
            model_path=str(self.model_path),
            metadata_path=str(self.metadata_path),
            preprocessing_path=str(self.preprocessing_path),
            metrics_path=str(self.metrics_path),
            trained_at=self.trained_at,
            deployed_at=datetime.now(timezone.utc).isoformat(),
            training_data_cutoff_timestamp=self.training_data_cutoff_timestamp,
            make_active=True,
        )
