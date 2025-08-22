import cv2
import numpy as np
import tensorflow as tf
import pickle

# Load model and test with simple shapes
model = tf.keras.models.load_model('asl_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Large detection area
    cv2.rectangle(frame, (50, 50), (450, 450), (0, 255, 0), 3)
    cv2.putText(frame, "Show number gesture", (60, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Extract and process ROI
    roi = frame[50:450, 50:450]
    roi_resized = cv2.resize(roi, (64, 64))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_normalized = roi_rgb.astype(np.float32) / 255.0
    
    # Predict
    prediction = model.predict(np.expand_dims(roi_normalized, axis=0), verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Always show the top prediction
    gesture = label_encoder.inverse_transform([predicted_class])[0]
    color = (0, 255, 0) if confidence > 0.15 else (0, 255, 255)  # Green if confident, yellow if not
    
    # Large number display
    cv2.putText(frame, gesture, (200, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 8, color, 15)
    
    # Show confidence
    cv2.putText(frame, f"{confidence:.2f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('ASL Numbers Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()