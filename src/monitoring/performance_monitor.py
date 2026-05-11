import sqlite3
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.config import settings
from src.logging.db import DatabaseManager


class PerformanceMonitor:
    def __init__(
        self,
        db_path=settings.DB_PATH,
        db_manager: DatabaseManager | None = None,
    ):
        self.db_path = db_path
        self.db_manager = db_manager or DatabaseManager(db_path=db_path)

    def load_prediction_logs(self) -> pd.DataFrame:
        active_deployment = self.db_manager.get_active_deployment()

        if active_deployment is None:
            query = """
            SELECT *
            FROM prediction_logs
            WHERE actual_label IS NOT NULL
            """

            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)

            return df

        query = """
        SELECT *
        FROM prediction_logs
        WHERE actual_label IS NOT NULL
        AND deployment_id = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(active_deployment["id"],),
            )

        return df

    def compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            raise ValueError("No labeled prediction logs found")

        y_true = df["actual_label"].astype(int)
        y_pred = df["prediction"].astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            "total_predictions": int(len(df)),
            "fraud_prediction_rate": float(df["prediction"].mean()),
            "average_fraud_probability": float(df["fraud_probability"].mean()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "model_version": str(df["model_version"].iloc[-1]),
            "deployment_id": (
                int(df["deployment_id"].iloc[-1])
                if pd.notna(df["deployment_id"].iloc[-1])
                else None
            ),
        }

        return metrics

    def run(self) -> dict[str, Any]:
        df = self.load_prediction_logs()
        metrics = self.compute_metrics(df)
        return metrics
