import pandas as pd
import requests

from src.config import settings

val_df = pd.read_csv(settings.VALIDATION_PATH)

DROP_COLS = settings.DROP_COLS

sample_row = val_df.iloc[249]
sample = sample_row.drop(columns=DROP_COLS).to_dict()
actual_label = sample_row[settings.TARGET_COL]

response = requests.post("http://localhost:8000/predict", json={"features" : sample, "actual_label": int(actual_label) })

print("Response status code:", response.status_code)
print("Response JSON:", response.json())