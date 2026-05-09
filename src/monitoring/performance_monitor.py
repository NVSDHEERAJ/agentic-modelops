import sqlite3
from typing import Any

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.config import settings

class PerformanceMonitor:
    def __init__(self, db_path = settings.DB_PATH):
        self.db_path = db_path
    
    def load_prediction_logs(self) -> pd.DataFrame:
        query = """
        SELECT * FROM prediction_logs
        WHERE actual_label IS NOT NULL"""

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)

        return df
    
    def compute_metrics(self, df : pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            raise ValueError("No labeled prediction logs found")
        
        y_true = df["actual_label"].astype(int)
        y_pred = df["prediction"].astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            "total_predictions" : int(len(df)),
            "fraud_prediction_rate" : float(df["prediction"].mean()),
            "average_fraud_probability" : float(df["fraud_probability"].mean()),
            "precision" : precision_score(y_true, y_pred, zero_division = 0),
            "recall" : recall_score(y_true, y_pred, zero_division = 0),
            "f1" : f1_score(y_true, y_pred, zero_division = 0),
            "true_negatives" : int(tn),
            "false_positives" : int(fp),
            "false_negatives" : int(fn),
            "true_positives" : int(tp)
        }

        return metrics
    
    def run(self) -> dict[str, Any]:
        df = self.load_prediction_logs()
        metrics = self.compute_metrics(df)
        return metrics