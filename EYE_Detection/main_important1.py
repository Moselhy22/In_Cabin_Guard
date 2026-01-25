import cv2 as cv
import mediapipe as mp
import time
import utils
import numpy as np

# variables
frame_counter = 0
CLOSED_EYES_FRAME = 3
CEF_COUNTER = 0
TOTAL_BLINKS = 0
FONTS = cv.FONT_HERSHEY_COMPLEX

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# camera object
camera = cv.VideoCapture(0)

# start time
start_time = time.time()

while True:
    frame_counter += 1
    ret, frame = camera.read()

    if not ret:
        break

    frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = utils.landmarksDetection(frame, results, False)
        ratio = utils.blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

        if ratio > 5.5:
            CEF_COUNTER += 1
        else:
            if CEF_COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                CEF_COUNTER = 0

        utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

        cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
        cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)

        right_coords = [mesh_coords[p] for p in RIGHT_EYE]
        left_coords = [mesh_coords[p] for p in LEFT_EYE]
        crop_right, crop_left = utils.eyesExtractor(frame, right_coords, left_coords)

        eye_position_right, _ = utils.positionEstimator(crop_right)
        utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2)

        eye_position_left, _ = utils.positionEstimator(crop_left)
        utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2)

        if eye_position_right == 'RIGHT' or eye_position_left == 'LEFT':
            if time.time() - start_time >= 5:
                print("This is an error. An accident may occur with the vehicle.")

    end_time = time.time() - start_time
    fps = frame_counter / end_time
    frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

    cv.imshow('frame', frame)
    key = cv.waitKey(2)
    if key == ord('q') or key == ord('Q'):
        break

cv.destroyAllWindows()
camera.release()