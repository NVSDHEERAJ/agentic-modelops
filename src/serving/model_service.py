import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import settings
from src.logging.db import DatabaseManager


class FraudModelService:
    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        self.db_manager = db_manager or DatabaseManager()
        self.db_manager.initialize_database()

        self.deployment_id: int | None = None
        self.deployed_at: str | None = None

        self.model_path: Path
        self.metadata_path: Path
        self.preprocessing_path: Path

        self.model: LGBMClassifier
        self.metadata: dict[str, Any]
        self.features: list[str]
        self.threshold: float
        self.model_version: str
        self.encoders: Dict[str, LabelEncoder]

        self.load_model()

    def load_model(self) -> None:
        active_deployment = self.db_manager.get_active_deployment()

        if active_deployment is None:
            self.model_path = settings.MODEL_PATH
            self.metadata_path = settings.METADATA_PATH
            self.preprocessing_path = settings.DEFAULT_PREPROCESSING_PATH

            self.deployment_id = None
            self.deployed_at = None
        else:
            self.model_path = Path(active_deployment["model_path"])
            self.metadata_path = Path(active_deployment["metadata_path"])
            self.preprocessing_path = Path(active_deployment["preprocessing_path"])

            self.deployment_id = int(active_deployment["id"])
            self.deployed_at = active_deployment["deployed_at"]

        self.model = joblib.load(self.model_path)

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.features = self.metadata["features"]
        self.threshold = float(self.metadata["threshold"])
        self.model_version = self.metadata.get(
            "model_version",
            settings.DEFAULT_MODEL_VERSION,
        )
        self.encoders = joblib.load(self.preprocessing_path)

    def prepare_input(self, features: dict[str, Any]) -> pd.DataFrame:
        row = pd.DataFrame([features])

        for col in self.features:
            if col not in row.columns:
                row[col] = None

        for col, encoder in self.encoders.items():
            row[col] = row[col].astype("string").fillna("missing")

            known_classes = set(encoder.classes_)

            row[col] = row[col].apply(lambda x: x if x in known_classes else "unknown")
            encoded_values = encoder.transform(row[col].astype(str))
            row[col] = pd.Series(encoded_values, index=row.index)

        numeric_cols = [
            col
            for col in self.features
            if col not in self.encoders
        ]

        for col in numeric_cols:
            row[col] = pd.to_numeric(row[col], errors="coerce")

        row[numeric_cols] = row[numeric_cols].fillna(-999)

        row = row[self.features]

        return row

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        row = self.prepare_input(features)

        probabilities = np.asarray(self.model.predict_proba(row))
        fraud_probability = float(probabilities[0][1])
        prediction = int(fraud_probability >= self.threshold)

        return {
            "prediction": prediction,
            "fraud_probability": fraud_probability,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "deployment_id": self.deployment_id,
            "deployed_at": self.deployed_at,
        }
    
    def reload_model(self) -> None:
        self.load_model()