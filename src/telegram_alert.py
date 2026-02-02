import asyncio
import logging
from telegram import Bot

def send_telegram_alert(bot_token: str, chat_id: str, message: str):
    """Send Telegram alert asynchronously."""
    async def _send():
        bot = Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message)
    try:
        asyncio.run(_send())
        logging.info("✅ Telegram alert sent.")
    except Exception as e:
        logging.error(f"❌ Telegram alert failed: {e}")