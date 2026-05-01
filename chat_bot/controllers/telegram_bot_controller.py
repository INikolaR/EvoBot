from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone
from assistant.pipeline.rag_service import RAGService
from chat_bot.services.history_service import HistoryService
from chat_bot.services.models import RequestModel

class TelegramBotController:
    def __init__(self, token: str, rag_service: RAGService, history_service: HistoryService):
        self.chat_type = 1
        
        self.token = token
        self.rag_service = rag_service
        self.history_service = history_service
        self.app = None

        self.start_message = "Привет! Напишите ваш вопрос."
        self.help_message = "Отправьте ваш вопрос одним сообщением, и я на него отвечу! Для сброса контекста используйте команду /reset, для вывода дополнительной информации обо мне - команду /about. По команде /help выводится это сообщение."
        self.reset_message = "Контекст сброшен, начинаем с чистого листа!"
        self.about_message = "Я - ИИ-консультант по настольной игре \"Эволюция\", разработаной Дмитрием Кнорре в 2010 году. Из-за большого объёма игровых правил у игроков часто возникают вопросы, для ответа на которые нужно перечитывать правила от начала до конца или даже обращаться к помощи других игроков на специальных форумах. Со мной же в этом нет необходимости: я с радостью предоставлю вам ответ на любой вопрос по игре меньше, чем за минуту!"
        self.unknown_message = "Извините, я не понимаю эту команду. Введите /help для получения подсказки."

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("reset", self.reset))
        self.app.add_handler(CommandHandler("about", self.about))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.app.add_handler(MessageHandler(filters.COMMAND, self.unknown))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(self.start_message)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.history_service.reset_context(user_id, self.chat_type)
        await update.message.reply_text(self.reset_message)

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(self.about_message)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(self.help_message)

    async def unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(self.unknown_message)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.message.text
        user_id = update.effective_user.id
        
        user_request_time = update.message.date
        if user_request_time.tzinfo is not None:
            user_request_time = user_request_time.astimezone()

        prev_context = self.history_service.get_context_for_llm(user_id, self.chat_type)

        response, context_docs = self.rag_service.get_response(text, [prev_context])
        
        response_time = datetime.now()

        request_model = RequestModel(
            user_id=user_id,
            chat_type=self.chat_type,
            user_text=text,
            model_response=response[0],
            rag_context="\n".join(context_docs[0]),
            request_time=user_request_time,
            response_time=response_time
        )
        self.history_service.add_request(request_model)
        
        await update.message.reply_text(response[0])

    def run(self):
        self.app = Application.builder().token(self.token).build()
        self.setup_handlers()
        self.app.run_polling()