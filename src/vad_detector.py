import torch
import numpy as np
import logging

class SileroVAD:
    def __init__(self, threshold=0.4, sampling_rate=16000, device=None):
        """
        Initialize Silero VAD model.
        
        Args:
            threshold (float): Speech probability threshold (0.0–1.0). Lower = more sensitive.
            sampling_rate (int): Must be 16000 Hz for Silero VAD.
            device (str): 'cuda' or 'cpu'
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        
        # Load Silero VAD model from torch.hub
        self.model, _ = torch.hub.load(
            'snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self.model.to(self.device)
        logging.info(f"✅ Silero VAD loaded on {self.device}.")

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if a 1D float32 audio array (16kHz) contains speech.
        Audio can be any length >= 512 samples (~32ms).
        Returns True if ANY 512-sample window has speech probability > threshold.
        """
        if len(audio) < 512:
            return False  # Silero VAD requires at least 512 samples at 16kHz

        # Convert to tensor and move to device
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        window_size = 512  # Fixed for 16kHz in Silero VAD
        has_speech = False

        with torch.no_grad():
            # Process in 512-sample chunks
            for i in range(0, len(audio_tensor), window_size):
                chunk = audio_tensor[i:i + window_size]
                # Pad last chunk if shorter than 512
                if len(chunk) < window_size:
                    chunk = torch.nn.functional.pad(chunk, (0, window_size - len(chunk)))
                # Run VAD inference
                speech_prob = self.model(chunk, self.sampling_rate).item()
                if speech_prob > self.threshold:
                    has_speech = True
                    break  # Early exit on first speech detection

        logging.debug(f"VAD: {'SPEECH' if has_speech else 'SILENCE'} in {len(audio)} samples")
        return has_speech