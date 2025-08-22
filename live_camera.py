import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque

class LiveASLRecognizer:
    def __init__(self, model_path='asl_model.h5', encoder_path='label_encoder.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.frame_buffer = deque(maxlen=5)
        self.prediction_history = deque(maxlen=10)
        
    def preprocess_frame(self, frame):
        # Extract larger hand region for better detection
        roi = frame[50:450, 50:450]  # Larger ROI extraction
        roi = cv2.resize(roi, (64, 64))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = roi.astype(np.float32) / 255.0
        return roi
    
    def predict_gesture(self, frame):
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if confidence > 0.05:  # Very low confidence threshold
            gesture = self.label_encoder.inverse_transform([predicted_class])[0]
            self.prediction_history.append(gesture)
            
            # Get most common prediction in recent history
            if len(self.prediction_history) >= 3:
                most_common = max(set(self.prediction_history), key=self.prediction_history.count)
                return most_common, confidence
        
        return None, confidence
    
    def run_camera(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw larger ROI rectangle
            cv2.rectangle(frame, (50, 50), (450, 450), (0, 255, 0), 3)
            cv2.putText(frame, "Place hand here", (60, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Predict gesture
            gesture, confidence = self.predict_gesture(frame)
            
            # Always show prediction
            prediction = model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            gesture = self.label_encoder.inverse_transform([predicted_class])[0]
            
            color = (0, 255, 0) if confidence > 0.15 else (0, 255, 255)
            cv2.putText(frame, gesture, (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 8, color, 15)
            cv2.putText(frame, f"{confidence:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('ASL Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recognizer = LiveASLRecognizer()
        recognizer.run_camera()
    except FileNotFoundError:
        print("Model files not found. Please train the model first by running train.py")