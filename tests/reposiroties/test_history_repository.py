import pytest
import sqlite3
import os
from chat_bot.repositories.history_repository import HistoryRepository
from chat_bot.repositories.entities import RequestEntity

def _fetch_last_record(cursor):
    cursor.execute("SELECT * FROM history ORDER BY request_id DESC LIMIT 1")
    return cursor.fetchone()

class TestHistoryRepository:
    class TestInitialization:
        def test_creates_db_file(self, db_path):
            assert not os.path.exists(db_path)
            HistoryRepository(db_path=db_path)
            assert os.path.exists(db_path)

        def test_creates_history_table(self, db_path):
            HistoryRepository(db_path=db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'")
            assert cursor.fetchone() is not None
            conn.close()

        def test_has_required_columns(self, db_path):
            HistoryRepository(db_path=db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(history)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()
            required = {
                "request_id", "user_text", "model_response", "user_id_hash",
                "interface_type", "user_request_time", "model_response_time"
            }
            assert required.issubset(columns)

    class TestAddRequest:
        def test_minimal(self, repository, db_connection):
            repository.add_request(RequestEntity(
                user_id="100", user_text="Привет", model_response="Ответ",
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["user_text"] == "Привет"
            assert rec["model_response"] == "Ответ"
            assert rec["user_id_hash"] == "100"
            assert rec["rag_context"] is None
            assert rec["prev_request_id"] is None

        def test_with_optional_fields(self, repository, db_connection):
            repository.add_request(RequestEntity(
                user_id=200, user_text="Вопрос", model_response="Ответ",
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["rag_context"] == '["doc1", "doc2"]'
            assert rec["prev_request_id"] is None

        def test_multiple_records(self, repository, db_connection):
            for i in range(1, 4):
                repository.add_request(RequestEntity(
                    user_id=200, user_text=f"Вопрос{i}", model_response=f"Ответ{i}",
                    rag_context='["doc1", "doc2"]', prev_request_id=None,
                    chat_type=1, request_time=0, response_time=0
                ))
            cursor = db_connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM history")
            assert cursor.fetchone()[0] == 3

        def test_special_characters(self, repository, db_connection):
            text = "Вопрос с 'кавычками' и \"двойными\""
            resp = "Ответ с ; DROP TABLE --"
            repository.add_request(RequestEntity(
                user_id=200, user_text=text, model_response=resp,
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["user_text"] == text
            assert rec["model_response"] == resp

        def test_unicode_support(self, repository, db_connection):
            text = "Привет, 🦎!"
            repository.add_request(RequestEntity(
                user_id=200, user_text=text, model_response=text,
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["user_text"] == text

    class TestConnectionManagement:
        def test_data_persists(self, db_path):
            repo = HistoryRepository(db_path=db_path)
            repo.add_request(RequestEntity(
                user_id=200, user_text="q", model_response="a",
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM history")
            assert cursor.fetchone()[0] == 1
            conn.close()

    class TestEdgeCases:
        def test_empty_strings(self, repository, db_connection):
            repository.add_request(RequestEntity(
                user_id=200, user_text="", model_response="",
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["user_text"] == ""
            assert rec["model_response"] == ""

        def test_large_payload(self, repository, db_connection):
            long_text = "A" * 10000
            repository.add_request(RequestEntity(
                user_id=200, user_text=long_text, model_response=long_text,
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["user_text"] == long_text

        def test_nullable_foreign_key(self, repository, db_connection):
            repository.add_request(RequestEntity(
                user_id=200, user_text="how?", model_response="that's how",
                rag_context='["doc1", "doc2"]', prev_request_id=None,
                chat_type=1, request_time=0, response_time=0
            ))
            rec = _fetch_last_record(db_connection.cursor())
            assert rec["prev_request_id"] is None