import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from assistant.pipeline.rag_service import RAGService
from chat_bot.services.history_service import HistoryService
from chat_bot.services.models import RequestModel
from datetime import datetime
import hashlib

class VKBotController:
    def __init__(self, token: str, group_id: int, rag_service: RAGService, history_service: HistoryService, salt: str):
        self.chat_type = 2
        self.salt = salt
        
        self.token = token
        self.group_id = group_id
        self.rag_service = rag_service
        self.history_service = history_service

        self.vk = vk_api.VkApi(token=self.token)
        self.api = self.vk.get_api()
        self.longpoll = VkBotLongPoll(self.vk, self.group_id)

        self.start_message = "Привет! Напишите ваш вопрос"
        self.help_message = "Отправьте ваш вопрос одним сообщением, и я на него отвечу! Для сброса контекста используйте команду /reset, для вывода дополнительной информации обо мне - команду /about. По команде /help выводится это сообщение"
        self.reset_message = "Контекст сброшен, начинаем с чистого листа! Напишите ваш вопрос"
        self.about_message = "Я - ИИ-консультант по настольной игре \"Эволюция\", разработаной Дмитрием Кнорре в 2010 году. Из-за большого объёма игровых правил у игроков часто возникают вопросы, для ответа на которые нужно перечитывать правила от начала до конца или даже обращаться к помощи других игроков на специальных форумах. Со мной же в этом нет необходимости: я с радостью предоставлю вам ответ на любой вопрос по игре меньше, чем за минуту!"
    
    def _send_message(self, peer_id: int, text: str):
        self.api.messages.send(
            peer_id=peer_id,
            message=text,
            random_id=0
        )

    def _get_anonymous_id(self, user_id: int) -> str:
        return hashlib.sha256(f"{user_id}_{self.salt}".encode()).hexdigest()

    def _handle_message(self, event):
        msg = event.object.message
        raw_text = msg.get("text", "").strip()
        text = raw_text.lower()
        peer_id = msg.get("peer_id")

        if text in ("/start"):
            self._send_message(peer_id, self.start_message)
        elif text in ("/help"):
            self._send_message(peer_id, self.help_message)
        elif text in ("/reset"):
            self.history_service.reset_context(self._get_anonymous_id(peer_id), self.chat_type)
            self._send_message(peer_id, self.reset_message)
        elif text in ("/about"):
            self._send_message(peer_id, self.about_message)
        else:
            user_id = self._get_anonymous_id(peer_id)
            vk_timestamp = msg.get("date")
            user_request_time = datetime.fromtimestamp(vk_timestamp)

            prev_context = self.history_service.get_context_for_llm(user_id, self.chat_type)

            response, context_docs = self.rag_service.get_response(raw_text, [prev_context])

            response_time = datetime.now()

            request_model = RequestModel(
                user_id=user_id,
                chat_type=self.chat_type,
                user_text=raw_text,
                model_response=response[0],
                rag_context="\n".join(context_docs[0]),
                request_time=user_request_time,
                response_time=response_time
            )

            self.history_service.add_request(request_model)
            
            self._send_message(peer_id, response[0])

    def run(self):
        for event in self.longpoll.listen():
            if event.type == VkBotEventType.MESSAGE_NEW:
                try:
                    self._handle_message(event)
                except Exception as e:
                    pass
