import os
import cv2
import numpy as np

def create_sample_data():
    # Create sample dataset structure
    base_path = "asl_alphabet_train"
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # ASL Numbers 0-9
    
    os.makedirs(base_path, exist_ok=True)
    
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        # Create 50 sample images per class
        for i in range(50):
            # Generate random colored rectangles as sample hand gestures
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            # Create hand-like shapes for each number
            if class_name == '0':
                # Closed fist - circle
                cv2.circle(img, (100, 100), 40, (220, 180, 140), -1)
                cv2.circle(img, (100, 100), 40, (0, 0, 0), 3)
            elif class_name == '1':
                # One finger up - vertical rectangle
                cv2.rectangle(img, (90, 60), (110, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            elif class_name == '2':
                # Two fingers - peace sign
                cv2.rectangle(img, (80, 60), (95, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (105, 60), (120, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            elif class_name == '3':
                # Three fingers
                cv2.rectangle(img, (75, 60), (85, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (95, 60), (105, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (115, 60), (125, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            elif class_name == '4':
                # Four fingers
                cv2.rectangle(img, (70, 60), (80, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (85, 60), (95, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (105, 60), (115, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (120, 60), (130, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            elif class_name == '5':
                # Open hand - all fingers
                cv2.rectangle(img, (65, 60), (75, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (80, 60), (90, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (95, 60), (105, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (110, 60), (120, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (125, 60), (135, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 30, (220, 180, 140), -1)
            elif class_name == '6':
                # Thumb and pinky
                cv2.rectangle(img, (70, 80), (85, 120), (220, 180, 140), -1)
                cv2.rectangle(img, (115, 60), (125, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            elif class_name == '7':
                # Thumb, index, middle
                cv2.rectangle(img, (70, 80), (85, 120), (220, 180, 140), -1)
                cv2.rectangle(img, (90, 60), (100, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (105, 60), (115, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            elif class_name == '8':
                # All except pinky
                cv2.rectangle(img, (70, 80), (85, 120), (220, 180, 140), -1)
                cv2.rectangle(img, (90, 60), (100, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (105, 60), (115, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (120, 60), (130, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            else:  # 9
                # All except index
                cv2.rectangle(img, (70, 80), (85, 120), (220, 180, 140), -1)
                cv2.rectangle(img, (105, 60), (115, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (120, 60), (130, 140), (220, 180, 140), -1)
                cv2.rectangle(img, (125, 60), (135, 140), (220, 180, 140), -1)
                cv2.circle(img, (100, 150), 25, (220, 180, 140), -1)
            
            # Save image
            filename = f"{class_name}_{i:03d}.jpg"
            cv2.imwrite(os.path.join(class_path, filename), img)
    
    print(f"ASL Numbers dataset created with {len(classes)} classes, 50 images each")
    print("Numbers:", classes)

if __name__ == "__main__":
    create_sample_data()