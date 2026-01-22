import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = "chat_history.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
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

    conn.commit()
    conn.close()

def add_request(
    user_request_time: datetime,
    model_response_time: datetime,
    telegram_user_id: int,
    user_text: str,
    model_response: str,
    rag_context: Optional[str] = None,
    prev_request_id: Optional[int] = None
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO history (
            user_request_time,
            model_response_time,
            telegram_user_id,
            prev_request_id,
            rag_context,
            user_text,
            model_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_request_time,
        model_response_time,
        telegram_user_id,
        prev_request_id,
        rag_context,
        user_text,
        model_response
    ))
    conn.commit()
    conn.close()
