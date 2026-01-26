from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2

from src.config import LANDMARKS_PATH

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARKS_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_driver_roi(frame):
    h, w = frame.shape[:2]
    return (int(w * 0.25), int(h * 0.25)), (int(w * 0.75), int(h * 0.75))

def detect_faces_in_roi(gray, roi):
    rects = detector(gray, 0)
    (x1, y1), (x2, y2) = roi
    return [r for r in rects if x1 <= r.left() and r.right() <= x2 and y1 <= r.top() and r.bottom() <= y2]

def process_frame_for_eyes(gray, rect):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    return ear, leftEye, rightEye