from vkbottle.bot import Bot, Message
from assistant.pipeline.rag_service import RAGService
from chat_bot.services.history_service import HistoryService

class VKBotController:
    def __init__(self, token: str, rag_service: RAGService, history_service: HistoryService):
        self.bot = Bot(token)
        self.rag_service = rag_service
        self.history_service = history_service
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.bot.on.message(text="!старт | /start | привет")
        async def start_handler(message: Message):
            await message.answer("Привет! Напишите ваш вопрос по игре «Эволюция».")
        
        @self.bot.on.message(text="!помощь | /help")
        async def help_handler(message: Message):
            await message.answer("Отправьте ваш вопрос одним сообщением, и я на него отвечу! Для сброса контекста используйте команду /reset, для вывода дополнительной информации обо мне - команду /about. По команде /help выводится это сообщение.")
        
        @self.bot.on.message(text="!сброс | /reset")
        async def reset_handler(message: Message):
            await message.answer("Контекст сброшен, начинаем с чистого листа!")
        
        @self.bot.on.message(text="!о себе | /about")
        async def about_handler(message: Message):
            await message.answer("Я - ИИ-консультант по настольной игре \"Эволюция\", разработаной Дмитрием Кнорре в 2010 году. Из-за большого объёма игровых правил у игроков часто возникают вопросы, для ответа на которые нужно перечитывать правила от начала до конца или даже обращаться к помощи других игроков на специальных форумах. Со мной же в этом нет необходимости: я с радостью предоставлю вам ответ на любой вопрос по игре меньше, чем за минуту!")
        
        @self.bot.on.message()
        async def main_handler(message: Message):
            response, context_docs = self.rag_service.get_response(message.text)
            self.history_service.add_request(
                user_id=message.from_id,
                user_text=message.text,
                model_response=response,
                rag_context=context_docs
            )
            await message.answer(response)
    
    def run(self):
        print("Starting async VK bot...")
        self.bot.run_forever()