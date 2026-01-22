from telegram import Update
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_experiments.pipeline.main_pipeline import get_rag_response
from datetime import datetime
from history.chat_history import init_database, add_request

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set!")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Напишите ваш вопрос")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Я могу отвечать на сообщения по игре Эволюция. Напишите ваш вопрос!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    telegram_user_id = update.effective_user.id
    user_request_time = datetime.now()
    response, rag_context = get_rag_response(text)
    model_response_time = datetime.now()
    prev_request_id = None
    add_request(
        user_request_time=user_request_time,
        model_response_time=model_response_time,
        telegram_user_id=telegram_user_id,
        user_text=text,
        model_response=response,
        rag_context=rag_context,
        prev_request_id=prev_request_id
    )
    await update.message.reply_text(response)

def main() -> None:
    print("starting bot...")
    init_database()
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()

if __name__ == '__main__':
    main()
