import sqlite3
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

class HistoryService:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_database()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_request_time DATETIME NOT NULL,
                    model_response_time DATETIME NOT NULL,
                    telegram_user_id INTEGER NOT NULL,
                    prev_request_id INTEGER,
                    rag_context TEXT,
                    user_text TEXT NOT NULL,
                    model_response TEXT NOT NULL,
                    FOREIGN KEY (prev_request_id) REFERENCES history(request_id)
                )
            ''')

    def add_request(self, user_id: int, user_text: str, model_response: str, 
                    rag_context: Optional[str] = None, prev_request_id: Optional[int] = None):
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO history (user_request_time, model_response_time, telegram_user_id, 
                                   prev_request_id, rag_context, user_text, model_response) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (now, now, user_id, prev_request_id, rag_context, user_text, model_response))
