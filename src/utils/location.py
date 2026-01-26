import requests
import geoip2.database
from src.config import GEOIP_DB_PATH

def get_location():
    try:
        ip = requests.get("https://api.ipify.org?format=json", timeout=5).json().get("ip", "127.0.0.1")
        with geoip2.database.Reader(GEOIP_DB_PATH) as reader:
            loc = reader.city(ip)
            return loc.location.latitude, loc.location.longitude
    except Exception as e:
        print(f"Location error: {e}")
        return None, None