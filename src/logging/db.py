import sqlite3
from pathlib import Path
from typing import Any

from src.config import settings


class DatabaseManager:
    def __init__(self, db_path=settings.DB_PATH):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def initialize_database(self):
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL UNIQUE,
                    model_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    preprocessing_path TEXT,
                    metrics_path TEXT,
                    trained_at TEXT,
                    deployed_at TEXT NOT NULL,
                    training_data_cutoff_timestamp TEXT,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    deployment_id INTEGER,
                    features_json TEXT NOT NULL,
                    fraud_probability REAL NOT NULL,
                    prediction INTEGER NOT NULL,
                    decision_threshold REAL NOT NULL,
                    actual_label INTEGER,
                    FOREIGN KEY (deployment_id) REFERENCES model_deployments(id)
                );
                """
            )

            self._add_column_if_missing(
                conn=conn,
                table_name="prediction_logs",
                column_name="deployment_id",
                column_definition="INTEGER",
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version
                ON prediction_logs(model_version);
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_deployment_id
                ON prediction_logs(deployment_id);
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
                ON prediction_logs(timestamp);
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_deployments_active
                ON model_deployments(is_active);
                """
            )

            conn.commit()

    def _add_column_if_missing(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        existing_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name});").fetchall()
        }

        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition};"
            )

    def get_active_deployment(self) -> dict[str, Any] | None:
        query = """
        SELECT *
        FROM model_deployments
        WHERE is_active = 1
        ORDER BY deployed_at DESC
        LIMIT 1;
        """

        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(query).fetchone()

        return dict(row) if row is not None else None

    def register_deployment(
        self,
        model_version: str,
        model_path: str,
        metadata_path: str,
        deployed_at: str,
        preprocessing_path: str | None = None,
        metrics_path: str | None = None,
        trained_at: str | None = None,
        training_data_cutoff_timestamp: str | None = None,
        make_active: bool = True,
    ) -> int:
        with self.get_connection() as conn:
            if make_active:
                conn.execute(
                    """
                    UPDATE model_deployments
                    SET is_active = 0
                    WHERE is_active = 1;
                    """
                )

            conn.execute(
                """
                INSERT INTO model_deployments (
                    model_version,
                    model_path,
                    metadata_path,
                    preprocessing_path,
                    metrics_path,
                    trained_at,
                    deployed_at,
                    training_data_cutoff_timestamp,
                    is_active,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    model_path = excluded.model_path,
                    metadata_path = excluded.metadata_path,
                    preprocessing_path = excluded.preprocessing_path,
                    metrics_path = excluded.metrics_path,
                    trained_at = excluded.trained_at,
                    deployed_at = excluded.deployed_at,
                    training_data_cutoff_timestamp = excluded.training_data_cutoff_timestamp,
                    is_active = excluded.is_active,
                    status = excluded.status;
                """,
                (
                    model_version,
                    model_path,
                    metadata_path,
                    preprocessing_path,
                    metrics_path,
                    trained_at,
                    deployed_at,
                    training_data_cutoff_timestamp,
                    int(make_active),
                    "active",
                ),
            )

            row = conn.execute(
                """
                SELECT id
                FROM model_deployments
                WHERE model_version = ?;
                """,
                (model_version,),
            ).fetchone()

            conn.commit()

            if row is None:
                raise RuntimeError("Failed to register model deployment")

            return int(row[0])
