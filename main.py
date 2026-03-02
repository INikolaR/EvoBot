import os
from telegram_bot.controllers.telegram_bot_controller import TelegramBotController
from telegram_bot.services.rag_service import RAGService
from telegram_bot.services.history_service import HistoryService

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set!")

    rag_service = RAGService(knowledge_base_path="research/knowledge-base-rules.txt")
    history_service = HistoryService(db_path="data/chat_history.db")
    
    bot = TelegramBotController(token=token, rag_service=rag_service, history_service=history_service)
    bot.run()

if __name__ == "__main__":
    main()