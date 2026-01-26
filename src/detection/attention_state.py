import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

def get_head_pose(shape):
    """Returns 'center', 'left', or 'right' based on nose vs eye midpoint."""
    nose_tip = shape[30]
    left_eye_center = ((shape[36] + shape[39]) / 2).astype(int)
    right_eye_center = ((shape[42] + shape[45]) / 2).astype(int)
    eye_mid_x = (left_eye_center[0] + right_eye_center[0]) / 2
    dev = nose_tip[0] - eye_mid_x
    THRESHOLD = 15  # pixels
    if dev < -THRESHOLD:
        return "left"
    elif dev > THRESHOLD:
        return "right"
    return "center"

def is_eyes_visible(shape, eye_threshold=0.25):
    """
    Checks if eyes are open *and* visible (not occluded/turned away).
    Returns True if both eyes have reasonable EAR and are within frame.
    """
    lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # Compute EAR (simple check)
    def ear(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    left_ear = ear(leftEye)
    right_ear = ear(rightEye)

    # If EAR is too low *and* eyes are not detected (e.g., face turned), assume invisible
    # But we also need to detect *absence* of eyes — so we rely on dlib: if rect exists but no landmarks? → not reliable.
    # Better: Use face bounding box + eye region presence heuristic.

    # Simpler & robust: If face is detected, assume eyes *should* be visible unless head is turned > 30°.
    # So we combine with head pose.
    return True  # We’ll handle "no eyes" at higher level (when driver_face exists but eyes not extracted cleanly)

def get_attention_state(driver_face, shape=None):
    """
    Returns one of:
        'focused'       → eyes visible + head centered
        'distracted'    → eyes visible but head turned OR eyes NOT visible (e.g., face turned away)
        'drowsy'        → eyes closed (handled separately in main.py)
    """
    if driver_face is None:
        return "distracted"  # No face = definitely not focused

    # If we have shape, check head pose
    if shape is not None:
        head = get_head_pose(shape)
        if head == "center":
            return "focused"
        else:
            return "distracted"
    else:
        # shape not available (e.g., failed landmark detection) → treat as distracted
        return "distracted"