import os
import threading
from chat_bot.controllers.telegram_bot_controller import TelegramBotController
from chat_bot.controllers.vk_bot_controller import VKBotController
from assistant.pipeline.rag_service import RAGService
from chat_bot.repositories.history_repository import HistoryRepository
from chat_bot.services.history_service import HistoryService

def _run_vk_bot(vk_bot: VKBotController):
    vk_bot.run()

def main():
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tg_token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set!")
    
    vk_token = os.getenv("VK_BOT_TOKEN")
    if not vk_token:
        raise ValueError("VK_BOT_TOKEN not set!")
    
    vk_group_id = os.getenv("VK_COMMUNITY_ID")
    if not vk_token:
        raise ValueError("VK_COMMUNITY_ID not set!")
    
    path = os.getenv("SQLITE_PATH")
    if not vk_token:
        raise ValueError("SQLITE_PATH not set!")
    
    salt = os.getenv("CONTROLLER_SALT")
    if not vk_token:
        raise ValueError("CONTROLLER_SALT not set!")

    rag_service = RAGService()
    history_repository = HistoryRepository(db_path=path)
    history_service = HistoryService(history_repository)
    
    tg_bot = TelegramBotController(
        token=tg_token,
        rag_service=rag_service,
        history_service=history_service,
        salt=salt
    )
    vk_bot = VKBotController(
        token=vk_token,
        group_id=vk_group_id,
        rag_service=rag_service,
        history_service=history_service,
        salt=salt
    )
    
    vk_thread = threading.Thread(target=_run_vk_bot, args=(vk_bot,), daemon=True)
    vk_thread.start()
    tg_bot.run()
    
    print("STARTED")

if __name__ == "__main__":
    main()