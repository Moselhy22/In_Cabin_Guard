import os
import time
import logging
import yaml
import torch
import numpy as np
from src.audio_listener import record_audio
from src.transcriber import WhisperTranscriber
from src.danger_detector import contains_danger_word
from src.telegram_alert import send_telegram_alert

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/voice_monitor.log"),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = "config/settings.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Config file missing! Copy 'config/settings_template.yaml' to '{config_path}' and fill it."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    keywords = config["danger_keywords"]
    tg_token = config["telegram"]["bot_token"]
    tg_chat_id = config["telegram"]["chat_id"]
    duration = config["audio"]["chunk_duration"]
    sample_rate = config["audio"]["sample_rate"]

    # Initialize Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    transcriber = WhisperTranscriber(model_name="base", device=device)

    logging.info("🟢 Starting live danger monitor... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Record audio
            audio = record_audio(duration=duration, target_sample_rate=sample_rate)
            if len(audio) == 0:
                time.sleep(1)
                continue

            # Transcribe
            text = transcriber.transcribe(audio)
            if text:
                logging.info(f"📝 Transcribed: '{text}'")
                if contains_danger_word(text, keywords):
                    alert_msg = "🚨 ALERT: Friend is in danger!"
                    logging.critical(alert_msg)
                    send_telegram_alert(tg_token, tg_chat_id, alert_msg)
            else:
                logging.debug("🔇 No speech detected.")

            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("🛑 Stopped by user.")
    except Exception as e:
        logging.exception(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()