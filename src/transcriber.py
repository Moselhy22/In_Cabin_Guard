import whisper
import torch
import numpy as np
import logging

class WhisperTranscriber:
    def __init__(self, model_name="base", device=None):
        """
        Initialize Whisper speech-to-text model.
        
        Args:
            model_name (str): Whisper model size ('tiny', 'base', 'small')
            device (str): 'cuda' or 'cpu'
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        logging.info(f"✅ Whisper '{model_name}' loaded on {device}.")

    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        Transcribe 16kHz audio to text.
        
        Args:
            audio_np (np.ndarray): 1D float32 array of audio samples at 16kHz
            
        Returns:
            str: Transcribed text (empty if failed or silent)
        """
        if len(audio_np) == 0:
            return ""
        try:
            audio_np = audio_np.astype(np.float32)
            result = self.model.transcribe(
                audio_np,
                fp16=False,  # GTX 1650 compatibility
                language="en",
                # Guide model toward emergency vocabulary
                initial_prompt="Emergency help gun knife attack police danger",
                # Allow repeated words like "gun gun gun"
                suppress_tokens=[],  # ← Critical: disables repetition suppression
                temperature=0.0,     # deterministic output
                no_speech_threshold=0.1  # more sensitive to quiet speech
            )
            text = result["text"].strip()
            return text
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return ""