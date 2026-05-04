import sqlite3
from contextlib import contextmanager
from chat_bot.repositories.entities import RequestEntity

class HistoryRepository:
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
                    user_id_hash TEXT NOT NULL,
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
                    user_id_hash TEXT NOT NULL,
                    chat_type_id INTEGER NOT NULL,
                    last_request_id INTEGER,
                    PRIMARY KEY (user_id_hash, chat_type_id)
                    FOREIGN KEY (last_request_id) REFERENCES history(request_id)
                );
            ''')

    def add_request(self, request_entity: RequestEntity):
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT last_request_id FROM users WHERE user_id_hash = ? AND chat_type_id = ?',
                (request_entity.user_id, request_entity.chat_type)
            )
            row = cursor.fetchone()
            prev_request_id = row[0] if row else None

            cursor = conn.execute('''
                INSERT INTO history (user_request_time, model_response_time, user_id_hash, chat_type_id,
                                   prev_request_id, rag_context, user_text, model_response) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (request_entity.request_time,
                  request_entity.response_time,
                  request_entity.user_id,
                  request_entity.chat_type,
                  prev_request_id,
                  request_entity.rag_context,
                  request_entity.user_text,
                  request_entity.model_response))
        
            new_id = cursor.lastrowid

            conn.execute('''
                INSERT INTO users (user_id_hash, chat_type_id, last_request_id) 
                VALUES (?, ?, ?) 
                ON CONFLICT(user_id_hash, chat_type_id) DO UPDATE SET last_request_id = excluded.last_request_id
            ''', (request_entity.user_id,
                  request_entity.chat_type,
                  new_id))

    def get_last_request_id(self, user_id: str, chat_type_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT is_context_active FROM users WHERE user_id = ? AND chat_type_id = ?',
                (user_id, chat_type_id)
            )
            row = cursor.fetchone()
            return bool(row[0]) if row else False

    def reset_context(self, user_id: str, chat_type_id: int) -> None:
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO users (user_id_hash, chat_type_id, last_request_id) 
                VALUES (?, ?, NULL) 
                ON CONFLICT(user_id, chat_type_id) DO UPDATE SET last_request_id = excluded.last_request_id
            ''', (user_id, chat_type_id))
    
    def get_context_for_llm(self, user_id: str, chat_type_id: int, max_messages: int = 20) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT last_request_id FROM users WHERE user_id_hash = ? AND chat_type_id = ?',
                (user_id, chat_type_id)
            )
            row = cursor.fetchone()

            if not row or row[0] is None:
                return []
            
            last_id = row[0]

        query = '''
            WITH RECURSIVE context_chain AS (
                SELECT request_id, prev_request_id, user_text, model_response, user_request_time
                FROM history
                WHERE request_id = ?
                
                UNION ALL
                
                SELECT h.request_id, h.prev_request_id, h.user_text, h.model_response, h.user_request_time
                FROM history h
                INNER JOIN context_chain cc ON h.request_id = cc.prev_request_id
            )
            SELECT user_text, model_response, user_request_time
            FROM context_chain
            ORDER BY user_request_time ASC
        '''
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, (last_id,))
            rows = cursor.fetchall()

        context = [
            {"user": r[0], "model": r[1]}
            for r in rows[-max_messages:]
        ]
        
        return context
