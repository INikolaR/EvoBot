import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from chat_bot.repositories.history_repository import HistoryRepository
from chat_bot.services.history_service import HistoryService
from chat_bot.controllers.telegram_bot_controller import TelegramBotController
from chat_bot.controllers.vk_bot_controller import VKBotController
from assistant.pipeline.rag_service import RAGService
from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.chunkers.fixed_chunker import FixedLengthChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.generators.hf_model_generator import HFModelGenerator
import torch

import sqlite3

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_history.db")

@pytest.fixture
def repository(db_path):
    return HistoryRepository(db_path=db_path)

@pytest.fixture
def db_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@pytest.fixture
def mock_rag_service():
    mock = MagicMock()
    mock.get_response.return_value = ("Тестовый ответ от ИИ", [{"doc": "context"}])
    return mock


@pytest.fixture
def mock_history_service():
    mock = MagicMock()
    mock.add_request = MagicMock()
    return mock


@pytest.fixture
def tg_controller(mock_rag_service, mock_history_service):
    from chat_bot.controllers.telegram_bot_controller import TelegramBotController
    
    return TelegramBotController(
        token="test_token",
        rag_service=mock_rag_service,
        history_service=mock_history_service,
        salt="salt"
    )


@pytest.fixture
def make_mocked_update():
    def _make_mocked_update(text: str, user_id: int = 12345, chat_id: int = 12345):

        mock_message = MagicMock()
        mock_message.text = text
        mock_message.message_id = 1
        mock_message.reply_text = AsyncMock()

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.type = "private"
        mock_message.chat = mock_chat
        
        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.first_name = "TestUser"
        mock_user.is_bot = False
        mock_message.from_user = mock_user

        from telegram import Update
        update = Update(update_id=1, message=mock_message)
        
        return update, mock_message.reply_text
    
    return _make_mocked_update

@pytest.fixture
def mock_context():
    from unittest.mock import MagicMock
    context = MagicMock()
    return context

@pytest.fixture
def mock_vk_api():
    with patch('chat_bot.controllers.vk_bot_controller.vk_api.VkApi') as mock_vk:
        vk_instance = MagicMock()
        api_mock = MagicMock()
        vk_instance.get_api.return_value = api_mock
        mock_vk.return_value = vk_instance
        yield vk_instance, api_mock


@pytest.fixture
def mock_longpoll():
    with patch('chat_bot.controllers.vk_bot_controller.VkBotLongPoll') as mock_lp:
        longpoll_instance = MagicMock()
        mock_lp.return_value = longpoll_instance
        yield longpoll_instance

@pytest.fixture
def vk_controller(mock_vk_api, mock_longpoll, mock_rag_service, mock_history_service):
    from chat_bot.controllers.vk_bot_controller import VKBotController
    
    controller = VKBotController(
        token="test_token",
        group_id=123456,
        rag_service=mock_rag_service,
        history_service=mock_history_service,
        salt="test_salt"
    )
    return controller


@pytest.fixture
def make_mocked_vk_event():
    def _make_event(
        text: str,
        peer_id: int = 100500,
        user_id: int = 777,
        timestamp: int = None,
        message_id: int = 999
    ):
        if timestamp is None:
            from datetime import datetime
            timestamp = int(datetime.now().timestamp())
        
        event = MagicMock()
        event.type = MagicMock()
        event.type.value = 1
        event.object.message = {
            "text": text,
            "peer_id": peer_id,
            "from_id": user_id,
            "date": timestamp,
            "id": message_id,
            "conversation_id": 1
        }
        return event
    return _make_event

@pytest.fixture(params=[
    pytest.param(RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10), id="RecursiveCharacterChunker"),
    pytest.param(FixedLengthChunker(chunk_size=50, chunk_overlap=10), id="FixedLengthChunker"),
])
def chunker(request):
    return request.param

@pytest.fixture
def embedder_factory():
    return HFModelEmbedderFactory()

@pytest.fixture
def mock_huggingface_embeddings():
    with patch('assistant.components.embedders.hf_model_embedder_factory.HuggingFaceEmbeddings') as MockHF:
        instance = MagicMock()
        
        instance.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        instance.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        
        MockHF.return_value = instance
        yield MockHF

@pytest.fixture
def generator_instance():
    with patch("assistant.components.generators.hf_model_generator.AutoTokenizer") as MockTokenizer, \
         patch("assistant.components.generators.hf_model_generator.AutoModelForCausalLM") as MockModel:
        
        tok_mock = MagicMock()
        MockTokenizer.from_pretrained.return_value = tok_mock
        tok_mock.eos_token = "</s>"
        tok_mock.eos_token_id = 2
        tok_mock.pad_token = "</s>"
        
        tok_mock.apply_chat_template.return_value = ["<|user|>\nREQUEST\n<|assistant|>"]
        
        def mock_tokenize(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 4), dtype=torch.long)
            }
        tok_mock.side_effect = mock_tokenize
        
        tok_mock.decode.return_value = "generated answer"

        model_mock = MagicMock()
        MockModel.from_pretrained.return_value = model_mock
        model_mock.device = "cpu"
        model_mock.eval.return_value = model_mock
        
        def mock_generate(input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            return torch.ones((batch_size, 7), dtype=torch.long)
        model_mock.generate.side_effect = mock_generate

        yield HFModelGenerator(hf_model_name="test/model", use_4bit=False)
