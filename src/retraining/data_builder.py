import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.logging.db import DatabaseManager

class RetrainingDataBuilder:
    def __init__(
            self,
            db_manager : DatabaseManager | None = None,
            raw_train_path : Path = settings.RAW_TRAIN_PATH,
            raw_validation_path : Path = settings.RAW_VAL_PATH
    ) -> None:
        self.db_manager = db_manager or DatabaseManager()
        self.raw_train_path = raw_train_path
        self.raw_validation_path = raw_validation_path

    def load_baseline_raw_data(self) -> pd.DataFrame:
        train_df = pd.read_csv(self.raw_train_path)
        val_df = pd.read_csv(self.raw_validation_path)

        return pd.concat([train_df, val_df], ignore_index = True)
    
    def load_labeled_production_data(
            self,
            deployment_id : int,
            cutoff_timestamp : str,
    ) -> pd.DataFrame:
        query = """
        SELECT features_json, actual_label
        FROM prediction_logs
        WHERE deployment_id = ?
        and actual_label IS NOT NULL
        AND timestamp <= ?;
        """

        with self.db_manager.get_connection() as conn:
            logs_df = pd.read_sql_query(
                query,
                conn,
                params = (deployment_id, cutoff_timestamp)
            )

        if logs_df.empty:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(logs_df["features_json"].apply(json.loads).tolist())
        features_df[settings.TARGET_COL] = logs_df["actual_label"].astype(int)

        return features_df
    
    def build_combined_raw_dataset(
            self,
            active_deployment : dict[str, Any],
            cutoff_timestamp : str,
            output_dir : Path
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        baseline_df = self.load_baseline_raw_data()

        production_df = self.load_labeled_production_data(
            deployment_id = int(active_deployment["id"]),
            cutoff_timestamp = cutoff_timestamp
        )

        combined_df = (
            pd.concat([baseline_df, production_df], ignore_index = True)
            if not production_df.empty
            else baseline_df
        )

        output_dir.mkdir(parents = True, exist_ok = True)
        combined_raw_path = output_dir / "raw_combined.csv"
        combined_df.to_csv(combined_raw_path, index = False)

        summary = {
            "raw_train_path": str(self.raw_train_path),
            "raw_validation_path": str(self.raw_validation_path),
            "combined_raw_path": str(combined_raw_path),
            "baseline_rows": int(len(baseline_df)),
            "labeled_production_rows": int(len(production_df)),
            "combined_rows": int(len(combined_df)),
            "training_data_cutoff_timestamp": cutoff_timestamp,
            "source_deployment_id": int(active_deployment["id"]),
            "source_model_version": active_deployment["model_version"],
        }

        return combined_df, summary