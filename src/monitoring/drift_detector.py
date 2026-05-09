import json
import sqlite3
from typing import Any

import joblib
import pandas as pd
from scipy.stats import ks_2samp

from src.config import settings

class DriftDetector:
    def __init__(
            self,
            reference_data_path = settings.VALIDATION_PATH,
            db_path = settings.DB_PATH,
            model_path = settings.MODEL_PATH,
            metadata_path = settings.METADATA_PATH,
            drop_columns = settings.DROP_COLS,
            p_value_threshold = 0.05,
            top_k_features : int = 30
    ) -> None:
        self.reference_data_path = reference_data_path
        self.db_path = db_path
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.drop_columns = drop_columns
        self.p_value_threshold = p_value_threshold
        self.top_k_features = top_k_features

    def load_reference_data(self) -> pd.DataFrame:
        return pd.read_csv(self.reference_data_path)
    
    def load_production_data(self) -> pd.DataFrame:
        query = """
        SELECT features_json
        FROM prediction_logs;
        """

        with sqlite3.connect(self.db_path) as conn:
            logs_df = pd.read_sql_query(query, conn)

        if logs_df.empty:
            raise ValueError("No prediction logs found in the database for drift detection")
        
        features = logs_df["features_json"].apply(json.loads)
        return pd.DataFrame(features.tolist())
    
    def load_model_features(self) -> list[str]:
        with open(self.metadata_path, "r", encoding = "utf-8") as f:
            metadata = json.load(f)
        
        return metadata["features"]
    
    def get_top_k_features(self) -> list[str]:
        model = joblib.load(self.model_path)
        model_features = self.load_model_features()

        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "feature" : model_features,
            "importance" : importances
        })

        feature_importance_df = feature_importance_df.sort_values(by = "importance", ascending = False)

        top_features = feature_importance_df.head(self.top_k_features)["feature"].tolist()

        return top_features
    
    def get_numeric_top_features(self, reference_df : pd.DataFrame, production_df : pd.DataFrame) -> list[str]:
        top_features = self.get_top_k_features()

        numeric_features_top = [
            col
            for col in top_features
            if col in reference_df.columns and col in production_df.columns
            and col not in self.drop_columns
            and pd.api.types.is_numeric_dtype(reference_df[col])
        ]

        return numeric_features_top
    
    def detect_numeric_feature_drift(
            self,
            reference_df : pd.DataFrame,
            production_df : pd.DataFrame,
            numeric_cols : list[str]
    ) -> list[dict[str, Any]]:
        drift_results = []

        for col in numeric_cols:
            reference_values = reference_df[col].dropna()
            production_values = production_df[col].dropna()

            if reference_values.empty or production_values.empty:
                continue

            ks_result = ks_2samp(reference_values, production_values)
            statistic = float(ks_result.statistic) # type: ignore
            p_value = float(ks_result.pvalue) # type: ignore

            drift_detected = p_value < self.p_value_threshold

            drift_results.append({
                "feature" : col,
                "ks_statistic" : statistic,
                "p_value" : float(p_value),
                "drift_detected" : bool(drift_detected),
                "reference_mean" : float(reference_values.mean()),
                "production_mean" : float(production_values.mean()),
                "reference_std" : float(reference_values.std()),
                "production_std" : float(production_values.std())
            })

        return drift_results
        
    def detect_prediction_drift(self) -> dict[str, Any]:
        query = """
        SELECT fraud_probability, prediction
        FROM prediction_logs;
        """

        with sqlite3.connect(self.db_path) as conn:
            logs_df = pd.read_sql_query(query, conn)

        if logs_df.empty:
            raise ValueError("No prediction logs found in the database for drift detection")
        
        return {
            "average_fraud_probability" :float(logs_df["fraud_probability"].mean()),
            "fraud_prediction_rate" : float(logs_df["prediction"].mean()),
            "total_predictions" : int(len(logs_df))
        }
    
    def run(self) -> dict[str, Any]:
        reference_df = self.load_reference_data()
        production_df = self.load_production_data()

        numeric_cols = self.get_numeric_top_features(reference_df, production_df)

        feature_drift_results = self.detect_numeric_feature_drift(reference_df, production_df, numeric_cols)

        drifted_features = [result["feature"]
                            for result in feature_drift_results
                            if result["drift_detected"]
                            ]
        
        prediction_drift = self.detect_prediction_drift()

        return {
            "drift_detected" : len(drifted_features) > 0,
            "feature_selection_model" : "top_k_model_importance",
            "top_k_features" : self.top_k_features,
            "num_features_checked": len(feature_drift_results),
            "num_drifted_features": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_drift": feature_drift_results,
            "prediction_drift": prediction_drift,
        }