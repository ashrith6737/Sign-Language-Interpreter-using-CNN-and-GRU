import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import mediapipe as mp

class EnhancedASLRecognizer:
    def __init__(self, model_path='asl_model.h5', encoder_path='label_encoder.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # MediaPipe for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.prediction_history = deque(maxlen=10)
        
    def extract_hand_roi(self, frame, landmarks):
        h, w, _ = frame.shape
        
        # Get bounding box coordinates with larger padding
        x_coords = [landmark.x * w for landmark in landmarks.landmark]
        y_coords = [landmark.y * h for landmark in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)) - 40, int(max(x_coords)) + 40
        y_min, y_max = int(min(y_coords)) - 40, int(max(y_coords)) + 40
        
        # Ensure coordinates are within frame bounds
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        # Extract and resize ROI
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            roi = cv2.resize(roi, (64, 64))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi.astype(np.float32) / 255.0
            return roi, (x_min, y_min, x_max, y_max)
        
        return None, None
    
    def predict_gesture(self, roi):
        if roi is None:
            return None, 0.0
            
        prediction = self.model.predict(np.expand_dims(roi, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if confidence > 0.2:
            gesture = self.label_encoder.inverse_transform([predicted_class])[0]
            self.prediction_history.append(gesture)
            
            if len(self.prediction_history) >= 3:
                most_common = max(set(self.prediction_history), key=self.prediction_history.count)
                return most_common, confidence
        
        return None, confidence
    
    def run_camera(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Enhanced ASL Recognition - Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands.process(rgb_frame)
            
            gesture_text = "No hand detected"
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract hand ROI and predict
                    roi, bbox = self.extract_hand_roi(frame, hand_landmarks)
                    gesture, confidence = self.predict_gesture(roi)
                    
                    if bbox:
                        x_min, y_min, x_max, y_max = bbox
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    if gesture:
                        gesture_text = f"Gesture: {gesture}"
                    else:
                        gesture_text = "Processing..."
            
            # Display results
            cv2.putText(frame, gesture_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Enhanced ASL Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recognizer = EnhancedASLRecognizer()
        recognizer.run_camera()
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
    except ImportError:
        print("MediaPipe not installed. Using basic version...")
        from live_camera import LiveASLRecognizer
        recognizer = LiveASLRecognizer()
        recognizer.run_camera()