import sqlite3
import pandas as pd

from src.config import settings

conn = sqlite3.connect(settings.DB_PATH)

df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 5", conn)

print(df)
conn.close()