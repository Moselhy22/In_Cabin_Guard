import whisper
import torch
import numpy as np
import logging

class WhisperTranscriber:
    def __init__(self, model_name="base", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        logging.info(f"✅ Whisper '{model_name}' loaded on {device}.")

    def transcribe(self, audio_np: np.ndarray) -> str:
        """Transcribe 16kHz float32 numpy array to text."""
        if len(audio_np) == 0:
            return ""
        try:
            # Ensure audio is 1D and float32
            audio_np = audio_np.astype(np.float32)
            result = self.model.transcribe(
                audio_np,
                fp16=False,  # GTX 1650 may not support fp16 well
                language="en"  # optional: force English for speed/accuracy
            )
            text = result["text"].strip()
            return text
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return ""