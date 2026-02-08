import cv2, time, asyncio, os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from src.config import EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, SOS_DELAY
from src.detection.legacy.eye_drowsiness import get_largest_face, detector
from src.alerts.alarm_player import AlarmPlayer
from src.alerts.telegram_notifier import send_sos_message, send_wake_up_notification
from src.utils.location import get_location
from imutils import face_utils
import dlib
from src.config import LANDMARKS_PATH
from scipy.spatial import distance as dist

# Load predictor
predictor = dlib.shape_predictor(LANDMARKS_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants
HEAD_TURN_THRESHOLD_PX = 15
HEAD_TURN_CONSEC_FRAMES = 30
DISTRACTED_TIMEOUT = 5  # seconds of distraction before alarm

cap = cv2.VideoCapture(0)
alarm_player = AlarmPlayer()

COUNTER_EYES = 0
COUNTER_DISTRACTED = 0
drowsiness_start_time = None
distraction_start_time = None
sos_sent = False

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        rects = detector(gray, 0)
        driver_face = get_largest_face(rects)

        eyes_open = True
        state = "focused"  # default

        if driver_face is not None:
            x1, y1 = driver_face.left(), driver_face.top()
            x2, y2 = driver_face.right(), driver_face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            try:
                # Try to get landmarks
                shape = predictor(gray, driver_face)
                shape = face_utils.shape_to_np(shape)

                # === EYES ===
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = (dist.euclidean(leftEye[1], leftEye[5]) + dist.euclidean(leftEye[2], leftEye[4])) / (2.0 * dist.euclidean(leftEye[0], leftEye[3]))
                rightEAR = (dist.euclidean(rightEye[1], rightEye[5]) + dist.euclidean(rightEye[2], rightEye[4])) / (2.0 * dist.euclidean(rightEye[0], rightEye[3]))
                ear = (leftEAR + rightEAR) / 2.0

                for eye in [cv2.convexHull(leftEye), cv2.convexHull(rightEye)]:
                    cv2.drawContours(frame, [eye], -1, (0, 255, 0), 1)

                # Head pose
                nose_tip = shape[30]
                left_eye_c = ((shape[36] + shape[39]) / 2).astype(int)
                right_eye_c = ((shape[42] + shape[45]) / 2).astype(int)
                eye_mid_x = (left_eye_c[0] + right_eye_c[0]) / 2
                nose_dev = nose_tip[0] - eye_mid_x

                if abs(nose_dev) > HEAD_TURN_THRESHOLD_PX:
                    state = "distracted"
                else:
                    state = "focused"

                # Drowsiness
                if ear < EYE_AR_THRESH:
                    eyes_open = False
                    COUNTER_EYES += 1
                    if drowsiness_start_time is None:
                        drowsiness_start_time = time.time()
                else:
                    COUNTER_EYES = 0
                    drowsiness_start_time = None

            except Exception as e:
                # Landmark detection failed → likely eyes not visible (head turned, looking down, etc.)
                print(f"[WARN] Landmark extraction failed: {e}")
                state = "distracted"
                COUNTER_EYES = 0  # reset drowsiness counter

        else:
            # No face detected → definitely not focused
            state = "distracted"

        # === STATE MACHINE ===
        if state == "distracted":
            COUNTER_DISTRACTED += 1
            if distraction_start_time is None:
                distraction_start_time = time.time()
            elapsed_dist = int(time.time() - distraction_start_time)
            cv2.putText(frame, f"ATTENTION ALERT! ({elapsed_dist}s)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if COUNTER_DISTRACTED >= HEAD_TURN_CONSEC_FRAMES:
                alarm_player.start()
        else:
            COUNTER_DISTRACTED = 0
            distraction_start_time = None

        # Drowsiness alert (higher priority)
        drowsy_alert = COUNTER_EYES >= EYE_AR_CONSEC_FRAMES
        if drowsy_alert:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alarm_player.start()

            if drowsiness_start_time and (time.time() - drowsiness_start_time) >= SOS_DELAY and not sos_sent:
                cv2.putText(frame, "SOS TRIGGERED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                lat, lon = get_location() or (37.7749, -122.4194)
                asyncio.run(send_sos_message(lat, lon))
                sos_sent = True

        # Wake-up logic
        if state == "focused" and eyes_open:
            alarm_player.stop()
            if sos_sent:
                asyncio.run(send_wake_up_notification())
                sos_sent = False

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")