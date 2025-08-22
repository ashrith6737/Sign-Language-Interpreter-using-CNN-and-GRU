import cv2
import numpy as np
import time
from collections import deque

def detect_fingers(roi):
    # Convert to HSV for better skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Skin color range
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Count skin pixels
    hand_pixels = np.sum(mask == 255)
    total_pixels = mask.shape[0] * mask.shape[1]
    hand_ratio = hand_pixels / total_pixels
    
    # Detect based on hand area for 1-5
    if hand_ratio < 0.05:
        return "?"
    elif hand_ratio < 0.15:
        return "1"
    elif hand_ratio < 0.3:
        return "2"
    elif hand_ratio < 0.45:
        return "3"
    elif hand_ratio < 0.6:
        return "4"
    else:
        return "5"

cap = cv2.VideoCapture(0)
gesture_history = deque(maxlen=10)
current_gesture = "1"
last_update = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Draw detection area
    cv2.rectangle(frame, (50, 50), (450, 450), (0, 255, 0), 3)
    cv2.putText(frame, "gesture_test.py - Show hand (1-5)", (60, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Extract ROI
    roi = frame[50:450, 50:450]
    
    # Detect gesture
    detected_gesture = detect_fingers(roi)
    gesture_history.append(detected_gesture)
    
    # Update display with delay (every 0.5 seconds)
    if time.time() - last_update > 0.5:
        if len(gesture_history) >= 5:
            # Use most common gesture in recent history
            current_gesture = max(set(gesture_history), key=gesture_history.count)
        last_update = time.time()
    
    # Display large number
    cv2.putText(frame, current_gesture, (200, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 20)
    
    # Show hand detection info
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    hand_pixels = np.sum(mask == 255)
    total_pixels = mask.shape[0] * mask.shape[1]
    hand_ratio = hand_pixels / total_pixels
    
    cv2.putText(frame, f"Hand: {hand_ratio:.3f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('gesture_test.py - Numbers 1-5', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()