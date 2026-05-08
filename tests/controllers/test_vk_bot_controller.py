import pytest


@pytest.mark.usefixtures("mock_vk_api", "mock_longpoll", "mock_rag_service", "mock_history_service")
class TestVKBotController:
    
    async def test_handle_start_command(self, vk_controller, make_mocked_vk_event, mock_vk_api):
        _, api_mock = mock_vk_api
        event = make_mocked_vk_event(text="/start")
        
        vk_controller._handle_message(event)
        
        api_mock.messages.send.assert_called_once_with(
            peer_id=100500,
            message=vk_controller.start_message,
            random_id=0
        )

    async def test_handle_help_command(self, vk_controller, make_mocked_vk_event, mock_vk_api):
        _, api_mock = mock_vk_api
        event = make_mocked_vk_event(text="/help")
        
        vk_controller._handle_message(event)
        
        api_mock.messages.send.assert_called_once_with(
            peer_id=100500,
            message=vk_controller.help_message,
            random_id=0
        )

    async def test_handle_reset_command(self, vk_controller, make_mocked_vk_event, mock_vk_api, mock_history_service):
        _, api_mock = mock_vk_api
        
        event = make_mocked_vk_event(text="/reset", peer_id=100500)
        vk_controller._handle_message(event)
        
        expected_user_id = vk_controller._get_anonymous_id(100500)
        mock_history_service.reset_context.assert_called_once_with(expected_user_id, vk_controller.chat_type)
        
        api_mock.messages.send.assert_called_once_with(
            peer_id=100500,
            message=vk_controller.reset_message,
            random_id=0
        )

    async def test_handle_about_command(self, vk_controller, make_mocked_vk_event, mock_vk_api):
        _, api_mock = mock_vk_api
        event = make_mocked_vk_event(text="/about")
        
        vk_controller._handle_message(event)
        
        api_mock.messages.send.assert_called_once()
        response_text = api_mock.messages.send.call_args.kwargs["message"]
        assert "Эволюция" in response_text
        assert "Дмитрием Кнорре" in response_text

    async def test_handle_regular_message_success(
        self, 
        vk_controller, 
        make_mocked_vk_event, 
        mock_vk_api, 
        mock_rag_service, 
        mock_history_service
    ):
        _, api_mock = mock_vk_api
        
        mock_history_service.get_context_for_llm.return_value = "контекст из истории"
        mock_rag_service.get_response.return_value = (
            ["Тестовый ответ от ИИ"],
            [["doc1", "doc2"]]
        )
        
        user_text = "Какие карты бывают в Эволюции?"
        event = make_mocked_vk_event(text=user_text, peer_id=100500, user_id=777)
        
        vk_controller._handle_message(event)
        
        expected_user_id = vk_controller._get_anonymous_id(100500)
        mock_rag_service.get_response.assert_called_once_with(
            user_text,
            ["контекст из истории"]
        )
        
        mock_history_service.add_request.assert_called_once()
        saved_request = mock_history_service.add_request.call_args.args[0]
        assert saved_request.user_id == expected_user_id
        assert saved_request.chat_type == vk_controller.chat_type
        assert saved_request.user_text == user_text
        assert saved_request.model_response == "Тестовый ответ от ИИ"
        assert "doc1" in saved_request.rag_context
        assert saved_request.request_time is not None
        assert saved_request.response_time is not None
        
        api_mock.messages.send.assert_called_once_with(
            peer_id=100500,
            message="Тестовый ответ от ИИ",
            random_id=0
        )

    async def test_handle_message_preserves_original_case_for_rag(
        self,
        vk_controller,
        make_mocked_vk_event,
        mock_vk_api,
        mock_rag_service
    ):
        _, api_mock = mock_vk_api
        mock_rag_service.get_response.return_value = (["OK"], [["ctx"]])
        
        original_text = "Какие Карты Бывают В Эволюции?"
        event = make_mocked_vk_event(text=original_text)
        
        vk_controller._handle_message(event)
        
        mock_rag_service.get_response.assert_called_once()
        call_args = mock_rag_service.get_response.call_args
        assert call_args.args[0] == original_text

    async def test_handle_unknown_command(self, vk_controller, make_mocked_vk_event, mock_vk_api):
        _, api_mock = mock_vk_api
        mock_rag_response = (["Ответ на вопрос"], [["контекст"]])
        vk_controller.rag_service.get_response.return_value = mock_rag_response
        
        event = make_mocked_vk_event(text="/unknown_command")
        vk_controller._handle_message(event)
        
        vk_controller.rag_service.get_response.assert_called_once()
        api_mock.messages.send.assert_called_once()
        assert api_mock.messages.send.call_args.kwargs["message"] == "Ответ на вопрос"
