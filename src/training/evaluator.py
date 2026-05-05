import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

class ModelEvaluator:
    def __init__(self, thresholds = None):
        self.thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.95, 0.01)

    def find_best_threshold(self, y_true, y_probs):
        best_threshold = max(
            self.thresholds,
            key = lambda t: f1_score(y_true, (y_probs >= t).astype(int), zero_division = 0)
        )
        return float(best_threshold)
    
    def evaluate(self, model, X_val, y_val):
        y_probs = model.predict(X_val)
        best_threshold = self.find_best_threshold(y_val, y_probs)
        y_pred = (y_probs >= best_threshold).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_val, y_probs)),
            "pr_auc": float(average_precision_score(y_val, y_probs)),
            "precision": float(precision_score(y_val, y_pred, zero_division = 0)),
            "recall": float(recall_score(y_val, y_pred, zero_division = 0)),
            "f1": float(f1_score(y_val, y_pred, zero_division = 0)),
            "best_threshold": best_threshold,
            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        }
        return metrics