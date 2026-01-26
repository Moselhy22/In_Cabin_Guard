# In-Cabin Guard

## v1.0 — Drowsiness Detection

Detects eye closure using EAR. Triggers alarm and Telegram SOS.

## Setup
1. Install Conda env:
   ```bash
   conda env create -f environment.yml
   conda activate in_cabin_guard

2. Download assets manually into assets/:

    shape_predictor_68_face_landmarks.dat → dlib models
    GeoLite2-City.mmdb → MaxMind
    alarm.mp3 → any short alarm sound

3. Add Telegram credentials to src/config.py

4. Run: python -m src.main