import json
import joblib
from pathlib import Path
from src.config import settings

class ModelRegistry:
    def __init__(
            self,
            model_path = settings.MODEL_PATH,
            metadata_path = settings.METADATA_PATH,
            metrics_path = settings.BASELINE_METRICS_PATH,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metrics_path = metrics_path

    def save_model(self, model):
        Path(self.model_path).parent.mkdir(parents = True, exist_ok = True)
        joblib.dump(model, self.model_path)

    def save_metadata(self, features, metrics):
        metadata = {
            "model_type" : "LighGBMClassifier",
            "features" : features,
            "threshold" : metrics.get("threshold"),
            "metrics" : metrics,
        }

        Path(self.metadata_path).parent.mkdir(parents = True, exist_ok = True)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent = 2)

    def save_metrics(self, metrics):
        Path(self.metrics_path).parent.mkdir(parents = True, exist_ok = True)
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent = 2)