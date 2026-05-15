import time
from typing import Any
import math

import pandas as pd
import requests

from src.config import settings

class ProductionTrafficSimulator:
    def __init__(
            self,
            data_path = settings.RAW_PROD_SIMULATION_PATH,
            api_url : str = settings.API_URL,
            target_col : str = settings.TARGET_COL,
            delay_seconds : float = 0.1
            ):
        self.data_path = data_path
        self.api_url = api_url
        self.target_col = target_col
        self.delay_seconds = delay_seconds

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)
    
    def build_payload(self, row: pd.Series) -> dict[str, Any]:
        actual_label = row[self.target_col]
        raw_features = row.drop(labels=settings.DROP_COLS).to_dict()

        features = {
            key: None if isinstance(value, float) and math.isnan(value) else value
            for key, value in raw_features.items()
        }

        return {
            "features": features,
            "actual_label": int(actual_label),
        }

    
    def send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(self.api_url, json = payload, timeout = 10)
        response.raise_for_status()
        return response.json()
    
    def run(self, max_rows: int | None = None, start_row: int = 0) -> None:
        df = self.load_data()

        if start_row < 0:
            raise ValueError("start_row must be greater than or equal to 0")

        if max_rows is None:
            df = df.iloc[start_row:]
        else:
            df = df.iloc[start_row:start_row + max_rows]

        success_count = 0
        failure_count = 0

        for index, row in df.iterrows():
            try:
                payload = self.build_payload(row)
                result = self.send_request(payload)

                success_count += 1
                if success_count % 100 == 0:
                    print(
                        f"Processed {success_count} rows | "
                        f"Latest prediction: {result['prediction']} | "
                        f"Fraud probability: {result['fraud_probability']:.4f}"
                    )

            except Exception as e:
                failure_count += 1
                print(f"Request failed, failed row index {index} : {str(e)}")

            time.sleep(self.delay_seconds)
        
        print(f"Simulation completed. Start row: {start_row}")
        print(f"Simulation completed. Total rows sent: {len(df)}")
        print(f"Success: {success_count}")
        print(f"Failures: {failure_count}")