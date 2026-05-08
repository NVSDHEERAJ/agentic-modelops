import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import settings


class FraudModelService:
    def __init__(
        self,
        model_path: Path = settings.MODEL_PATH,
        metadata_path: Path = settings.METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path

        self.model: LGBMClassifier
        self.metadata: dict[str, Any]
        self.features: list[str]
        self.threshold: float
        self.model_version: str
        self.encoders : Dict[str, LabelEncoder]

        self.load_model()

    def load_model(self) -> None:
        self.model = joblib.load(self.model_path)

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.features = self.metadata["features"]
        self.threshold = float(self.metadata["threshold"])
        self.model_version = self.metadata.get("model_version", "baseline_v1")
        self.encoders = joblib.load("models/preprocessing/label_encoders.pkl")

    def prepare_input(self, features: dict[str, Any]) -> pd.DataFrame:
        row = pd.DataFrame([features])

        for col in self.features:
            if col not in row.columns:
                row[col] = None

        # Handle Categorical columns
        for col, encoder in self.encoders.items():
            row[col] = row[col].astype("string").fillna("missing")

            known_classes = set(encoder.classes_)

            row[col] = row[col].apply(lambda x: x if x in known_classes else "unknown")
            encoded_values = encoder.transform(row[col].astype(str))
            row[col] = pd.Series(encoded_values, index=row.index)

        # Handle Numerical columns - fill missing values with -999
        row = row.fillna(-999)

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
        }