import os
import time
import logging
import yaml
import numpy as np
import torch
from collections import deque
from src.audio_listener import record_audio
from src.transcriber import WhisperTranscriber
from src.danger_detector import contains_danger_word
from src.telegram_alert import send_telegram_alert
from src.vad_detector import SileroVAD

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/voice_monitor_vad.log"),
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
    sample_rate = config["audio"]["sample_rate"]

    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad = SileroVAD(threshold=0.5, sampling_rate=sample_rate, device=device)
    transcriber = WhisperTranscriber(model_name="base", device=device)

    # Buffer to store recent audio (we'll record in small chunks)
    buffer_duration = 2.0  # seconds to keep in buffer
    chunk_duration = 0.5   # record in 500ms chunks
    buffer_samples = int(buffer_duration * sample_rate)
    chunk_samples = int(chunk_duration * sample_rate)
    
    audio_buffer = deque(maxlen=buffer_samples)

    logging.info("🟢 Starting VAD-enabled danger monitor... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Record short chunk
            chunk = record_audio(duration=chunk_duration, target_sample_rate=sample_rate)
            if len(chunk) == 0:
                time.sleep(0.1)
                continue

            # Add to buffer
            audio_buffer.extend(chunk)

            # Check if current chunk contains speech
            if vad.is_speech(chunk):
                logging.info("🗣️ Speech detected! Transcribing buffered audio...")
                
                # Transcribe full buffer
                buffered_audio = np.array(audio_buffer)
                text = transcriber.transcribe(buffered_audio)
                
                if text:
                    logging.info(f"📝 Transcribed: '{text}'")
                    if contains_danger_word(text, keywords):
                        alert_msg = "🚨 ALERT: Friend is in danger!"
                        logging.critical(alert_msg)
                        send_telegram_alert(tg_token, tg_chat_id, alert_msg)
                else:
                    logging.debug("🔇 Transcription returned empty.")
                
                # Clear buffer after transcription to avoid re-processing
                audio_buffer.clear()

            time.sleep(0.01)  # small delay

    except KeyboardInterrupt:
        logging.info("🛑 Stopped by user.")
    except Exception as e:
        logging.exception(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()