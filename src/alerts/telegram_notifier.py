from telegram import Bot
import asyncio
import re
from src.config import BOT_TOKEN, CHAT_ID

def escape_markdown(text):
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

async def send_sos_message(lat, lon, condition="Drowsiness"):
    link = f"https://www.google.com/maps?q={lat},{lon}"
    msg = f"SOS ALERT\\!\nDriver may be experiencing drowsiness\\.\nLocation: [Click here]({link})"
    for _ in range(3):
        try:
            await Bot(token=BOT_TOKEN).send_message(chat_id=CHAT_ID, text=msg, parse_mode="MarkdownV2")
            print("SOS sent!")
            return
        except Exception as e:
            print(f"SOS failed: {e}")

async def send_wake_up_notification():
    try:
        msg = "DRIVER AWAKE\\!\nThe driver has opened their eyes\\."
        await Bot(token=BOT_TOKEN).send_message(chat_id=CHAT_ID, text=msg, parse_mode="MarkdownV2")
        print("Wake-up sent!")
    except Exception as e:
        print(f"Wake-up failed: {e}")