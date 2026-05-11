import json
from datetime import datetime, timezone
from typing import Any

from src.logging.db import DatabaseManager

class PredictionLogger:
    def __init__(self, db_manager : DatabaseManager):
        self.db_manager = db_manager
    
    def log_prediction(
            self,
            features : dict[str, Any],
            fraud_probability : float,
            prediction : int,
            threshold : float,
            model_version : str,
            deployment_id: int |None = None,
            actual_label : int | None = None
    ) -> None:
        insert_query = """
        INSERT INTO prediction_logs (
            timestamp,
            model_version,
            deployment_id,
            features_json,
            fraud_probability,
            prediction,
            decision_threshold,
            actual_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
        
        values = (
            datetime.now(timezone.utc).isoformat(),
            model_version,
            deployment_id,
            json.dumps(features),
            fraud_probability,
            prediction,
            threshold,
            actual_label
        )

        with self.db_manager.get_connection() as conn:
            conn.execute(insert_query, values)
            conn.commit()