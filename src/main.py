import cv2, time, asyncio, os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from src.config import EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, SOS_DELAY
from src.detection.eye_drowsiness import get_driver_roi, detect_faces_in_roi, process_frame_for_eyes
from src.alerts.alarm_player import AlarmPlayer
from src.alerts.telegram_notifier import send_sos_message, send_wake_up_notification
from src.utils.location import get_location

cap = cv2.VideoCapture(0)
alarm_player = AlarmPlayer()
COUNTER = 0
drowsiness_start_time = None
sos_sent = False

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        roi = get_driver_roi(frame)
        cv2.rectangle(frame, roi[0], roi[1], (0,255,0), 2)

        rects = detect_faces_in_roi(gray, roi)
        eyes_open = True

        for rect in rects:
            ear, leftEye, rightEye = process_frame_for_eyes(gray, rect)
            for eye in [cv2.convexHull(leftEye), cv2.convexHull(rightEye)]:
                cv2.drawContours(frame, [eye], -1, (0,255,0), 1)

            if ear < EYE_AR_THRESH:
                eyes_open = False
                COUNTER += 1
                if drowsiness_start_time is None:
                    drowsiness_start_time = time.time()
                elapsed = int(time.time() - drowsiness_start_time)
                cv2.putText(frame, f"Eyes Closed: {elapsed} sec", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    alarm_player.start()

                if elapsed >= SOS_DELAY and not sos_sent:
                    cv2.putText(frame, "SOS TRIGGERED!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    lat, lon = get_location() or (37.7749, -122.4194)
                    asyncio.run(send_sos_message(lat, lon))
                    sos_sent = True
            else:
                COUNTER = 0

        if eyes_open:
            COUNTER = 0
            drowsiness_start_time = None
            alarm_player.stop()
            if sos_sent:
                asyncio.run(send_wake_up_notification())
                sos_sent = False
            cv2.putText(frame, "Eyes Open: 0 sec", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")