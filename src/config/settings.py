from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "current"
REPORT_DIR = BASE_DIR / "reports" / "metrics"
DB_PATH = BASE_DIR / "data" / "logs" / "predictions.db"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
PROD_SIMULATION_PATH = DATA_DIR / "production_simulation.csv"

MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
BASELINE_METRICS_PATH = REPORT_DIR / "baseline_metrics.json"

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

DROP_COLS = [
    TARGET_COL,
    ID_COL,
    TIME_COL
]

RANDOM_STATE = 42

API_URL = "http://localhost:8000/predict"