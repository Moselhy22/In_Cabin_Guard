import threading
import time
import subprocess
import os
from src.config import ALARM_PATH

class AlarmPlayer:
    def __init__(self):
        self.is_playing = False
        self.stop_flag = False
        self.process = None

    def _play_loop(self):
        self.is_playing = True
        print("Alarm started.")
        if not os.path.exists(ALARM_PATH):
            print(f"Alarm file missing: {ALARM_PATH}")
            self.is_playing = False
            return

        start_time = time.time()
        while time.time() - start_time < 7:
            if self.stop_flag: break
            self.process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", ALARM_PATH],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.process.wait()

        while self.is_playing and not self.stop_flag:
            self.process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", ALARM_PATH],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(0.5)
            if self.process.poll() is None:
                self.process.terminate()

        self.is_playing = False
        self.stop_flag = False
        print("Alarm stopped.")

    def start(self):
        if not self.is_playing:
            self.stop_flag = False
            threading.Thread(target=self._play_loop, daemon=True).start()

    def stop(self):
        self.stop_flag = True
        self.is_playing = False
        if self.process and self.process.poll() is None:
            self.process.terminate()