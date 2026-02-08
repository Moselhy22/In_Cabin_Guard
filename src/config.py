import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

ALARM_PATH = os.path.join(ASSETS_DIR, "audio", "alarm.mp3")
LANDMARKS_PATH = os.path.join(ASSETS_DIR, "models", "dlib/shape_predictor_68_face_landmarks.dat")
GEOIP_DB_PATH = os.path.join(ASSETS_DIR, "geoip", "GeoLite2-City.mmdb")

# Detection thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# Alert timing
ALARM_DURATION = 15
SOS_DELAY = 20

# Telegram (fill these locally — keep empty in repo)
BOT_TOKEN = "8102584813:AAGQuXN2H0zJT0nt4s9BXEJb05M9lttJKjY"
CHAT_ID = "1575296072"