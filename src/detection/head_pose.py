import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

def get_head_pose(shape):
    """
    Estimate head pose from 68-point landmarks.
    Returns:
        - 'center' if head is facing forward
        - 'left' if head turned left
        - 'right' if head turned right
    """
    # Get key points
    nose_tip = shape[30]
    left_eye_center = ((shape[36] + shape[39]) / 2).astype(int)
    right_eye_center = ((shape[42] + shape[45]) / 2).astype(int)

    # Calculate horizontal deviation
    eye_midpoint_x = (left_eye_center[0] + right_eye_center[0]) / 2
    nose_deviation = nose_tip[0] - eye_midpoint_x

    # Thresholds (adjust based on your face size)
    THRESHOLD = 15  # pixels — adjust if needed

    if nose_deviation < -THRESHOLD:
        return "left"
    elif nose_deviation > THRESHOLD:
        return "right"
    else:
        return "center"

def is_head_turned(shape, threshold=15):
    """
    Returns True if head is turned left or right.
    """
    pose = get_head_pose(shape)
    return pose in ["left", "right"]