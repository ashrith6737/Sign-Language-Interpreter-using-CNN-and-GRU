import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class ASLDataProcessor:
    def __init__(self, data_path, img_size=(64, 64)):
        self.data_path = data_path
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        images, labels = [], []
        
        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_name)
        
        images = np.array(images, dtype=np.float32) / 255.0
        labels = self.label_encoder.fit_transform(labels)
        labels = to_categorical(labels)
        
        return images, labels
    
    def get_classes(self):
        return self.label_encoder.classes_