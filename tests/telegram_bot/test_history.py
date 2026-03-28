import pytest
import sqlite3
import os
from datetime import datetime
from telegram_bot.services.history_service import HistoryService


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_history.db")


@pytest.fixture
def service(temp_db_path):
    return HistoryService(db_path=temp_db_path)


@pytest.fixture
def conn(temp_db_path):
    c = sqlite3.connect(temp_db_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def _get_last_record(cursor):
    cursor.execute("SELECT * FROM history ORDER BY request_id DESC LIMIT 1")
    return cursor.fetchone()


class TestInit:

    def test_db_file_created(self, temp_db_path):
        assert not os.path.exists(temp_db_path)
        HistoryService(db_path=temp_db_path)
        assert os.path.exists(temp_db_path)

    def test_table_exists(self, temp_db_path):
        HistoryService(db_path=temp_db_path)
        c = sqlite3.connect(temp_db_path)
        cur = c.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'")
        assert cur.fetchone() is not None
        c.close()

    def test_required_columns_exist(self, temp_db_path):
        HistoryService(db_path=temp_db_path)
        c = sqlite3.connect(temp_db_path)
        cur = c.cursor()
        cur.execute("PRAGMA table_info(history)")
        cols = {row[1] for row in cur.fetchall()}
        c.close()
        
        required = {'request_id', 'user_text', 'model_response', 'telegram_user_id', 
                   'user_request_time', 'model_response_time'}
        assert required.issubset(cols)


class TestAddRequest:

    def test_add_minimal(self, service, conn):
        service.add_request(
            user_id=100,
            user_text="Привет",
            model_response="Ответ"
        )
        rec = _get_last_record(conn.cursor())
        assert rec is not None
        assert rec['user_text'] == "Привет"
        assert rec['model_response'] == "Ответ"
        assert rec['telegram_user_id'] == 100
        assert rec['rag_context'] is None
        assert rec['prev_request_id'] is None

    def test_add_with_optional(self, service, conn):
        service.add_request(
            user_id=200,
            user_text="Вопрос",
            model_response="Ответ",
            rag_context='["doc1", "doc2"]',
            prev_request_id=42
        )
        rec = _get_last_record(conn.cursor())
        assert rec['rag_context'] == '["doc1", "doc2"]'
        assert rec['prev_request_id'] == 42

    def test_timestamps_stored(self, service, conn):
        before = datetime.now()
        service.add_request(300, "q", "a")
        after = datetime.now()
        
        rec = _get_last_record(conn.cursor())
        stored = datetime.fromisoformat(rec['user_request_time'])
        assert before <= stored <= after

    def test_multiple_records(self, service, conn):
        service.add_request(1, "q1", "a1")
        service.add_request(1, "q2", "a2")
        service.add_request(2, "q3", "a3")
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM history")
        assert cur.fetchone()[0] == 3
        
        cur.execute("SELECT COUNT(*) FROM history WHERE telegram_user_id = 1")
        assert cur.fetchone()[0] == 2

    def test_special_characters(self, service, conn):
        text = "Вопрос с 'кавычками' и \"двойными\""
        resp = "Ответ с ; DROP TABLE --"
        service.add_request(999, text, resp)
        
        rec = _get_last_record(conn.cursor())
        assert rec['user_text'] == text
        assert rec['model_response'] == resp
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM history")
        assert cur.fetchone()[0] >= 1

    def test_unicode(self, service, conn):
        text = "Привет! 🦎 Эволюция: 生物"
        service.add_request(777, text, text)
        rec = _get_last_record(conn.cursor())
        assert rec['user_text'] == text


class TestConnectionManagement:

    def test_connection_closed(self, temp_db_path):
        svc = HistoryService(db_path=temp_db_path)
        svc.add_request(1, "q", "a")
        c = sqlite3.connect(temp_db_path)
        cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM history")
        assert cur.fetchone()[0] == 1
        c.close()

    def test_rollback_on_error(self, service, conn):
        service.add_request(1, "ok", "ok")
        
        try:
            with service._get_connection() as c:
                cur = c.cursor()
                cur.execute(
                    "INSERT INTO history (telegram_user_id) VALUES (?)",
                    (999,)
                )
        except sqlite3.IntegrityError:
            pass

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM history")
        assert cur.fetchone()[0] == 1


class TestEdgeCases:

    def test_empty_strings(self, service, conn):
        service.add_request(555, "", "")
        rec = _get_last_record(conn.cursor())
        assert rec['user_text'] == ""
        assert rec['model_response'] == ""

    def test_null_rag_context(self, service, conn):
        service.add_request(333, "q", "a", rag_context=None)
        rec = _get_last_record(conn.cursor())
        assert rec['rag_context'] is None

    def test_large_text(self, service, conn):
        long = "A" * 10000
        service.add_request(888, long, long)
        rec = _get_last_record(conn.cursor())
        assert rec['user_text'] == long

    def test_foreign_key_nullable(self, service, conn):
        service.add_request(444, "q", "a", prev_request_id=None)
        rec = _get_last_record(conn.cursor())
        assert rec['prev_request_id'] is None