import pytest
from assistant.pipeline.rag_service import RAGService
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.mark.asyncio
class TestController:
    async def test_start(self, tg_controller, make_mocked_update):
        update, mock_reply = make_mocked_update("/start")
        await tg_controller.start(update, None)
        mock_reply.assert_called_once_with("Привет! Напишите ваш вопрос.")

    async def test_reset(self, tg_controller, make_mocked_update):
        update, mock_reply = make_mocked_update("/reset")
        await tg_controller.reset(update, None)
        mock_reply.assert_called_once_with("Контекст сброшен, начинаем с чистого листа!")

    async def test_about(self, tg_controller, make_mocked_update):
        update, mock_reply = make_mocked_update("/about")
        await tg_controller.about(update, None)
        mock_reply.assert_called_once()
        response_text = mock_reply.call_args.args[0]
        assert "Эволюция" in response_text

    async def test_help(self, tg_controller, make_mocked_update):
        update, mock_reply = make_mocked_update("/help")
        await tg_controller.help(update, None)
        mock_reply.assert_called_once()

    async def test_handle_message_success(self, tg_controller, make_mocked_update, mock_rag_service, mock_history_service):
        mock_history_service.get_context_for_llm.return_value = "контекст из истории"
        mock_rag_service.get_response.return_value = (
            ["Тестовый ответ от ИИ"],
            [["doc1", "doc2"]]
        )

        update, mock_reply = make_mocked_update("Какие карты бывают в Эволюции?")
        await tg_controller.handle_message(update, None)

        mock_rag_service.get_response.assert_called_once_with(
            "Какие карты бывают в Эволюции?",
            ["контекст из истории"]
        )
        mock_history_service.add_request.assert_called_once()
        mock_reply.assert_called_once_with("Тестовый ответ от ИИ")