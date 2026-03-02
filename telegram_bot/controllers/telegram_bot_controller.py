from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_experiments.pipeline.rag_service import RAGService
from telegram_bot.services.history_service import HistoryService

class TelegramBotController:
    def __init__(self, token: str, rag_service: RAGService, history_service: HistoryService):
        self.token = token
        self.rag_service = rag_service
        self.history_service = history_service
        self.app = None

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("reset", self.reset))
        self.app.add_handler(CommandHandler("about", self.about))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Привет! Напишите ваш вопрос")

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Контекст сброшен, начинаем с чистого листа!")

    async def about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Я - ИИ-консультант по настольной игре «Эволюция»...")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Отправьте ваш вопрос одним сообщением...")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.message.text
        user_id = update.effective_user.id
        
        response, context_docs = self.rag_service.get_response(text)
        
        self.history_service.add_request(
            user_id=user_id,
            user_text=text,
            model_response=response,
            rag_context=context_docs
        )
        
        await update.message.reply_text(response)

    def run(self):
        print("Starting bot...")
        self.app = Application.builder().token(self.token).build()
        self.setup_handlers()
        self.app.run_polling()