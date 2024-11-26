import cv2
import numpy as np
import time
from detector import ObjectDetector

# Initialize object detector (YOLO)
detector = ObjectDetector()

# List of objects to recognize
objects_to_find = ['book', 'cup', 'bottle']

# Initialize variables
score = 0
time_limit = 30  # Game time in seconds
start_time = time.time()

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = detector.detect_objects(frame)

    # Draw bounding boxes and check for the objects to find
    for label, confidence, bbox in detections:
        if label in objects_to_find:
            x, y, w, h = bbox
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            score += 1  # Increment score when an object is found

    # Display score and timer
    elapsed_time = time.time() - start_time
    remaining_time = max(0, time_limit - elapsed_time)
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Time Left: {int(remaining_time)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Object Recognition Game', frame)

    # Exit on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # End game when time runs out
    if remaining_time == 0:
        print(f"Game Over! Final Score: {score}")
        break

cap.release()
cv2.destroyAllWindows()
