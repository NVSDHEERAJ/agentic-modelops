import json
from typing import Any

import pandas as pd

from src.config import settings
from src.logging.db import DatabaseManager


class RetrainingDataBuilder:
    def __init__(self, db_manager: DatabaseManager | None = None):
        self.db_manager = db_manager or DatabaseManager()

    def load_original_data(self) -> pd.DataFrame:
        train_df = pd.read_csv(settings.TRAIN_PATH)
        validation_df = pd.read_csv(settings.VALIDATION_PATH)

        return pd.concat([train_df, validation_df], ignore_index=True)

    def load_labeled_production_data(
        self,
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

        with self.db_manager.get_connection() as conn:
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

    def build_combined_dataset(
        self,
        active_deployment: dict[str, Any],
        cutoff_timestamp: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        original_df = self.load_original_data()

        production_df = self.load_labeled_production_data(
            deployment_id=int(active_deployment["id"]),
            cutoff_timestamp=cutoff_timestamp,
        )

        combined_df = (
            pd.concat([original_df, production_df], ignore_index=True)
            if not production_df.empty
            else original_df
        )

        summary = {
            "original_rows": int(len(original_df)),
            "labeled_production_rows": int(len(production_df)),
            "combined_rows": int(len(combined_df)),
            "training_data_cutoff_timestamp": cutoff_timestamp,
        }

        return combined_df, summary