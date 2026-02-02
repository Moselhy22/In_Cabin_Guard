import sounddevice as sd
import numpy as np
import torch
import torchaudio
import logging

def record_audio(duration: float, target_sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from default mic at a supported rate (e.g., 48000),
    then resample to target_sample_rate (e.g., 16000 for Whisper).
    """
    # Common native rates: 44100, 48000
    native_rate = 48000
    channels = 1

    try:
        logging.info(f"🎙️ Recording {duration}s at {native_rate} Hz...")
        audio = sd.rec(
            int(duration * native_rate),
            samplerate=native_rate,
            channels=channels,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()  # shape: (samples,)

        # Resample to target rate (16kHz) using torchaudio
        if native_rate != target_sample_rate:
            audio_tensor = torch.from_numpy(audio).float()
            resampled = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=native_rate,
                new_freq=target_sample_rate
            )
            audio = resampled.numpy()

        logging.debug(f"✅ Resampled to {target_sample_rate} Hz. Shape: {audio.shape}")
        return audio

    except Exception as e:
        logging.error(f"Microphone error: {e}")
        return np.array([])