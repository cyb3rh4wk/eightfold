import json
import sqlite3
from typing import Dict, Any

class Persistence:
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_state(self, session_id: str, state: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO sessions (session_id, state) VALUES (?, ?)",
            (session_id, json.dumps({"history": state.get("history", [])}))
        )
        conn.commit()
        conn.close()

    def load_state(self, session_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        return json.loads(result[0]) if result else {"history": []}