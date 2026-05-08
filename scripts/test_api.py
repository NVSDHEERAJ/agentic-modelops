import pandas as pd
import requests

from src.config import settings

val_df = pd.read_csv(settings.VALIDATION_PATH)

DROP_COLS = settings.DROP_COLS

sample = val_df.drop(columns=DROP_COLS).iloc[10].to_dict()

response = requests.post("http://localhost:8000/predict", json={"features" : sample})

print("Response status code:", response.status_code)
print("Response JSON:", response.json())