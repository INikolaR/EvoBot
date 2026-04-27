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
                    user_id INTEGER NOT NULL,
                    chat_type_id INTEGER NOT NULL,
                    prev_request_id INTEGER,
                    rag_context TEXT,
                    user_text TEXT NOT NULL,
                    model_response TEXT NOT NULL,
                    FOREIGN KEY (prev_request_id) REFERENCES history(request_id)
                );
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER NOT NULL,
                    chat_type_id INTEGER NOT NULL,
                    is_context_active INTEGER NOT NULL,
                    PRIMARY KEY (user_id, chat_type_id)
                );
            ''')

    def add_request(self, user_id: int, chat_type: int, user_text: str, model_response: str,
                    request_time: datetime, response_time: datetime,
                    rag_context: Optional[str] = None, prev_request_id: Optional[int] = None):
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO history (user_request_time, model_response_time, user_id, chat_type_id,
                                   prev_request_id, rag_context, user_text, model_response) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (request_time, response_time, user_id, chat_type, prev_request_id, rag_context, user_text, model_response))

    def set_context_active(self, user_id: int, chat_type_id: int, is_active: bool) -> None:
        val = 1 if is_active else 0
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO users (user_id, chat_type_id, is_context_active) 
                VALUES (?, ?, ?) 
                ON CONFLICT(user_id, chat_type_id) DO UPDATE SET is_context_active = excluded.is_context_active
            ''', (user_id, chat_type_id, val))

    def get_context_active(self, user_id: int, chat_type_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT is_context_active FROM users WHERE user_id = ? AND chat_type_id = ?',
                (user_id, chat_type_id)
            )
            row = cursor.fetchone()
            return bool(row[0]) if row else False
