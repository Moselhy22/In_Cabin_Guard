import cv2
import numpy as np
import time

# Load the Haar cascade for eyes detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variables
prev_eye_positions = [None, None]
start_time = None
error_duration = 5  # Time duration in seconds for continuous eye movement
error_threshold = 10  # Threshold for detecting significant eye movement

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

    # Update previous eye positions
    prev_eye_positions = eyes

    # Check if eyes are detected
    if len(eyes) > 0:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= error_duration:
            display_error_message()
            break
    else:
        start_time = None  # Reset start time if eyes are not detected

    # Draw rectangles around the detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()