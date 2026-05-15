import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import settings


class CandidatePreprocessor:
    def __init__(
        self,
        data_output_dir: Path,
        artifact_output_dir: Path,
        numeric_missing_value: int = -999,
    ) -> None:
        self.data_output_dir = data_output_dir
        self.artifact_output_dir = artifact_output_dir
        self.numeric_missing_value = numeric_missing_value

        self.label_encoders: dict[str, LabelEncoder] = {}
        self.preprocessing_metadata: dict[str, Any] = {}

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if column not in settings.DROP_COLS
        ]

    def _infer_categorical_columns(self, X_train: pd.DataFrame) -> list[str]:
        return [
            column
            for column in X_train.columns
            if (
                X_train[column].dtype == "object"
                or str(X_train[column].dtype).startswith("string")
                or str(X_train[column].dtype) == "category"
            )
        ]

    def _fit_label_encoders(
        self,
        X_train: pd.DataFrame,
        categorical_columns: list[str],
    ) -> None:
        self.label_encoders = {}

        for column in categorical_columns:
            values = X_train[column].astype("string").fillna("missing")

            if "unknown" not in set(values.dropna().astype(str).unique()):
                values = pd.concat(
                    [
                        values,
                        pd.Series(["unknown"], index=[-1], dtype="string"),
                    ]
                )

            encoder = LabelEncoder()
            encoder.fit(values.astype(str))
            self.label_encoders[column] = encoder

    def _transform_with_label_encoders(
        self,
        X: pd.DataFrame,
        categorical_columns: list[str],
    ) -> pd.DataFrame:
        transformed = X.copy()

        for column in categorical_columns:
            encoder = self.label_encoders[column]
            known_classes = set(encoder.classes_)

            values = transformed[column].astype("string").fillna("missing")
            values = values.apply(lambda value: value if value in known_classes else "unknown")

            encoded_values = encoder.transform(values.astype(str))
            transformed[column] = pd.Series(
                encoded_values,
                index=transformed.index,
            )
            
        return transformed

    def _fill_numeric_missing_values(
        self,
        X: pd.DataFrame,
        categorical_columns: list[str],
    ) -> pd.DataFrame:
        transformed = X.copy()

        numeric_columns = [
            column
            for column in transformed.columns
            if column not in categorical_columns
        ]

        transformed[numeric_columns] = transformed[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        transformed[numeric_columns] = transformed[numeric_columns].fillna(
            self.numeric_missing_value
        )

        return transformed

    def split_fit_transform(
        self,
        combined_raw_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
        feature_columns = self._feature_columns(combined_raw_df)

        X = combined_raw_df[feature_columns].copy()
        y = combined_raw_df[settings.TARGET_COL].astype(int)

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=settings.RANDOM_STATE,
            stratify=y,
        )

        categorical_columns = self._infer_categorical_columns(X_train_raw)
        numeric_columns = [
            column
            for column in feature_columns
            if column not in categorical_columns
        ]

        self._fit_label_encoders(X_train_raw, categorical_columns)

        X_train = self._transform_with_label_encoders(
            X_train_raw,
            categorical_columns,
        )
        X_val = self._transform_with_label_encoders(
            X_val_raw,
            categorical_columns,
        )

        X_train = self._fill_numeric_missing_values(X_train, categorical_columns)
        X_val = self._fill_numeric_missing_values(X_val, categorical_columns)

        X_train = X_train[feature_columns]
        X_val = X_val[feature_columns]

        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_output_dir.mkdir(parents=True, exist_ok=True)

        train_output = X_train.copy()
        train_output[settings.TARGET_COL] = y_train.values

        validation_output = X_val.copy()
        validation_output[settings.TARGET_COL] = y_val.values

        train_path = self.data_output_dir / "train.csv"
        validation_path = self.data_output_dir / "validation.csv"
        encoders_path = self.artifact_output_dir / "label_encoders.pkl"
        preprocessing_metadata_path = (
            self.artifact_output_dir / "preprocessing_metadata.json"
        )

        train_output.to_csv(train_path, index=False)
        validation_output.to_csv(validation_path, index=False)

        joblib.dump(self.label_encoders, encoders_path)

        self.preprocessing_metadata = {
            "features": feature_columns,
            "target": settings.TARGET_COL,
            "drop_columns": settings.DROP_COLS,
            "categorical_columns": categorical_columns,
            "numeric_columns": numeric_columns,
            "numeric_missing_value": self.numeric_missing_value,
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "feature_count": int(len(feature_columns)),
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "label_encoders_path": str(encoders_path),
            "preprocessing_metadata_path": str(preprocessing_metadata_path),
        }

        with open(preprocessing_metadata_path, "w", encoding="utf-8") as file:
            json.dump(self.preprocessing_metadata, file, indent=2)

        return X_train, y_train, X_val, y_val, self.preprocessing_metadata