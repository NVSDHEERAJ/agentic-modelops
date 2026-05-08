import sqlite3
from pathlib import Path

from src.config import settings

class DatabaseManager:
    def __init__(self, db_path = settings.DB_PATH):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents = True, exist_ok = True)

    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def initialize_database(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            features_json TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            prediction INTEGER NOT NULL,    
            decision_threshold REAL NOT NULL,
            actual_label INTEGER
            );
            """
        
        with self.get_connection() as conn:
            conn.execute(create_table_query)
            conn.commit()