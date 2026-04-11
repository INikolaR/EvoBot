# tests/conftest.py
import pytest
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Добавляем корень проекта в PATH для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update, User, Message, Chat
from telegram.ext import ContextTypes


@pytest.fixture
def mock_rag_service():
    """Мокаем RAGService"""
    mock = MagicMock()
    mock.get_response.return_value = ("Тестовый ответ от ИИ", [{"doc": "context"}])
    return mock


@pytest.fixture
def mock_history_service():
    """Мокаем HistoryService"""
    mock = MagicMock()
    mock.add_request = MagicMock()  # синхронный метод
    return mock


@pytest.fixture
def controller(mock_rag_service, mock_history_service):
    """Создаёт экземпляр контроллера с моками"""
    from telegram_bot.controllers.telegram_bot_controller import TelegramBotController
    
    return TelegramBotController(
        token="test_token",
        rag_service=mock_rag_service,
        history_service=mock_history_service
    )


@pytest.fixture
def make_mocked_update():
    """
    Создаёт Update с полностью замокированным message.
    Возвращает кортеж: (update, mock_reply_text)
    """
    def _make_mocked_update(text: str, user_id: int = 12345, chat_id: int = 12345):
        # Создаём МОК вместо реального Message
        mock_message = MagicMock()
        mock_message.text = text
        mock_message.message_id = 1
        mock_message.reply_text = AsyncMock()
        
        # Мокаем chat и from_user
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.type = "private"
        mock_message.chat = mock_chat
        
        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.first_name = "TestUser"
        mock_user.is_bot = False
        mock_message.from_user = mock_user
        
        # Создаём Update с моком
        from telegram import Update
        update = Update(update_id=1, message=mock_message)
        
        return update, mock_message.reply_text
    
    return _make_mocked_update

@pytest.fixture
def mock_context():
    """Безопасный мок для CallbackContext"""
    from unittest.mock import MagicMock
    context = MagicMock()
    # Добавьте нужные атрибуты, если они используются:
    # context.bot = MagicMock()
    # context.user_data = {}
    return context