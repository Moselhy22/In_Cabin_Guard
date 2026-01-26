from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2

from src.config import LANDMARKS_PATH

# Load models once
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARKS_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_largest_face(rects):
    """
    Returns the face rectangle with the largest area.
    Assumes the largest face is the driver (closest to camera).
    """
    if not rects:
        return None
    return max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

def process_frame_for_eyes(gray, rect):
    """
    Extract eye landmarks and compute EAR for a given face rectangle.
    """
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    return ear, leftEye, rightEye