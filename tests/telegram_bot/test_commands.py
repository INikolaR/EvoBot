import pytest

@pytest.mark.asyncio
async def test_start_command(controller, make_mocked_update):
    update, mock_reply = make_mocked_update("/start")
    
    await controller.start(update, None)
    
    mock_reply.assert_called_once_with("Привет! Напишите ваш вопрос")


@pytest.mark.asyncio
async def test_reset_command(controller, make_mocked_update):
    update, mock_reply = make_mocked_update("/reset")
    
    await controller.reset(update, None)
    
    mock_reply.assert_called_once_with("Контекст сброшен, начинаем с чистого листа!")


@pytest.mark.asyncio
async def test_about_command(controller, make_mocked_update):
    update, mock_reply = make_mocked_update("/about")
    
    await controller.about(update, None)
    
    mock_reply.assert_called_once()
    response = mock_reply.call_args[0][0]
    assert "Эволюция" in response


@pytest.mark.asyncio
async def test_help_command(controller, make_mocked_update):
    update, mock_reply = make_mocked_update("/help")
    
    await controller.help_command(update, None)
    
    mock_reply.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_success(
    controller, make_mocked_update, mock_rag_service, mock_history_service
):
    update, mock_reply = make_mocked_update("Какие карты бывают в Эволюции?")
    
    await controller.handle_message(update, None)

    mock_rag_service.get_response.assert_called_once_with(
        "Какие карты бывают в Эволюции?"
    )

    mock_history_service.add_request.assert_called_once()

    mock_reply.assert_called_once_with("Тестовый ответ от ИИ")


@pytest.mark.asyncio
async def test_handle_message_rag_error(
    controller, make_mocked_update, mock_rag_service, mock_history_service
):
    mock_rag_service.get_response.side_effect = Exception("RAG failed")
    
    update, mock_reply = make_mocked_update("Сложный вопрос")

    with pytest.raises(Exception, match="RAG failed"):
        await controller.handle_message(update, None)