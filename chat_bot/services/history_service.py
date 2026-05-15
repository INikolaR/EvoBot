from chat_bot.repositories.history_repository import HistoryRepository
from chat_bot.services.models import RequestModel
from chat_bot.repositories.entities import RequestEntity

class HistoryService:
    def __init__(self, history_repository: HistoryRepository):
        self.history_repository = history_repository

    def add_request(self, request_model: RequestModel) -> None:
        request_entity = RequestEntity(request_model.user_id,
                                       request_model.chat_type,
                                       request_model.user_text,
                                       request_model.model_response,
                                       request_model.request_time,
                                       request_model.response_time,
                                       request_model.rag_context,
                                       request_model.prev_request_id)
        self.history_repository.add_request(request_entity)

    def reset_context(self, user_id: str, chat_type_id: int) -> None:
        self.history_repository.reset_context(user_id, chat_type_id)
    
    def get_context_for_llm(self, user_id: str, chat_type_id: int, max_messages: int = 20) -> list[dict]:
        context = self.history_repository.get_context_for_llm(user_id, chat_type_id, max_messages)
        chat_template_context = []
        for c in context:
            chat_template_context.append({"role" : "user", "content" : c["user"]})
            chat_template_context.append({"role" : "assistant", "content" : c["model"]})
        return chat_template_context
