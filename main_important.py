# https://github.com/Asadullah-Dal17/Eyes-Position-Estimator-Mediapipe.git

import cv2
import time

# Load the Haar cascade for eyes detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variables
start_time = None
error_duration = 5  # Time duration in seconds for continuous eye movement

def display_error_message():
    print("This is an error. An accident may occur with the vehicle.")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if eyes are detected
    if len(eyes) > 0:
        # Reset start time when eyes are detected
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= error_duration:
            display_error_message()
            start_time = None  # Reset start time after displaying the error message

    else:
        start_time = None  # Reset start time if eyes are not detected

    # Draw rectangles around the detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    # Check if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()