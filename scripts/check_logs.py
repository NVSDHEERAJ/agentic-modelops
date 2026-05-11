import sqlite3

import pandas as pd

from src.config import settings
from src.logging.db import DatabaseManager


def main() -> None:
    db_manager = DatabaseManager()
    db_manager.initialize_database()

    active_deployment = db_manager.get_active_deployment()

    print("Active deployment:")
    print(active_deployment)

    with sqlite3.connect(settings.DB_PATH) as conn:
        deployments_df = pd.read_sql_query(
            """
            SELECT *
            FROM model_deployments
            ORDER BY deployed_at DESC
            """,
            conn,
        )

        logs_df = pd.read_sql_query(
            """
            SELECT
                id,
                timestamp,
                model_version,
                deployment_id,
                fraud_probability,
                prediction,
                decision_threshold,
                actual_label
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT 10
            """,
            conn,
        )

    print("\nDeployments:")
    print(deployments_df)

    print("\nLatest prediction logs:")
    print(logs_df)


if __name__ == "__main__":
    main()
