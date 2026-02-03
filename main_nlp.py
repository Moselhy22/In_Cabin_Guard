import os
import time
import logging
import yaml
import numpy as np
import torch
from collections import deque
from src.audio_listener import record_audio
from src.transcriber import WhisperTranscriber
from src.nlp_classifier import DistressClassifier
from src.telegram_alert import send_telegram_alert
from src.vad_detector import SileroVAD

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/voice_monitor_hybrid.log"),
        logging.StreamHandler()
    ]
)

def load_config():
    config_path = "config/settings.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file missing! Copy 'config/settings_template.yaml' to '{config_path}' and fill it.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Critical danger keywords (case-insensitive, whole-word)
CRITICAL_KEYWORDS = {"gun", "knife", "bomb", "help", "emergency", "attack", "police"}

def contains_critical_keyword(text: str) -> bool:
    """Check for high-priority danger words (even if repeated)."""
    if not text:
        return False
    text_lower = text.lower()
    for word in CRITICAL_KEYWORDS:
        if word in text_lower:
            logging.warning(f"🚨 CRITICAL KEYWORD DETECTED: '{word}' in '{text}'")
            return True
    return False

def main():
    config = load_config()
    tg_token = config["telegram"]["bot_token"]
    tg_chat_id = config["telegram"]["chat_id"]
    sample_rate = config["audio"]["sample_rate"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad = SileroVAD(threshold=0.4, sampling_rate=sample_rate, device=device)
    transcriber = WhisperTranscriber(model_name="base", device=device)
    classifier = DistressClassifier(device=device)

    chunk_duration = 0.3
    max_buffer_duration = 5.0
    silence_duration = 1.0

    speech_buffer = deque()
    is_speaking = False
    last_speech_time = time.time()

    logging.info("🟢 Starting HYBRID danger monitor (keywords + NLP)... Press Ctrl+C to stop.")

    try:
        while True:
            chunk = record_audio(duration=chunk_duration, target_sample_rate=sample_rate)
            if len(chunk) == 0:
                time.sleep(0.01)
                continue

            has_speech = vad.is_speech(chunk)
            current_time = time.time()

            if has_speech:
                speech_buffer.extend(chunk)
                is_speaking = True
                last_speech_time = current_time
            else:
                if is_speaking and current_time - last_speech_time >= silence_duration:
                    if len(speech_buffer) > 0:
                        audio_to_transcribe = np.array(speech_buffer)
                        text = transcriber.transcribe(audio_to_transcribe)
                        if text.strip():
                            logging.info(f"📝 Transcribed: '{text}'")

                            # 🔥 HYBRID TRIGGER: keyword OR distress
                            if contains_critical_keyword(text) or classifier.is_distress(text, threshold=0.6):
                                alert_msg = "🚨 ALERT: Friend is in danger!"
                                logging.critical(alert_msg)
                                send_telegram_alert(tg_token, tg_chat_id, alert_msg)

                        speech_buffer.clear()
                        is_speaking = False
                    else:
                        is_speaking = False

                elif is_speaking:
                    # Keep buffering short silences within utterance
                    speech_buffer.extend(chunk)

            # Prevent buffer overflow
            if len(speech_buffer) > int(max_buffer_duration * sample_rate):
                logging.warning("⚠️ Utterance too long — forcing transcription")
                audio_to_transcribe = np.array(speech_buffer)
                text = transcriber.transcribe(audio_to_transcribe)
                if text.strip():
                    logging.info(f"📝 Transcribed (long): '{text}'")
                    if contains_critical_keyword(text) or classifier.is_distress(text, threshold=0.6):
                        alert_msg = "🚨 ALERT: Friend is in danger!"
                        logging.critical(alert_msg)
                        send_telegram_alert(tg_token, tg_chat_id, alert_msg)
                speech_buffer.clear()
                is_speaking = False

            time.sleep(0.01)

    except KeyboardInterrupt:
        logging.info("🛑 Stopped by user.")
    except Exception as e:
        logging.exception(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()