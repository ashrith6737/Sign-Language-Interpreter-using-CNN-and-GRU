import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GRU, TimeDistributed, Reshape

class ASLModel:
    def __init__(self, input_shape=(64, 64, 3), num_classes=29, sequence_length=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
    def build_cnn_gru_model(self):
        model = Sequential([
            # CNN Feature Extractor
            TimeDistributed(Conv2D(32, (3, 3), activation='relu'), 
                          input_shape=(self.sequence_length, *self.input_shape)),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Flatten()),
            
            # GRU for temporal modeling
            GRU(128, return_sequences=True),
            GRU(64),
            Dropout(0.5),
            
            # Classification layers
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_simple_cnn(self):
        # Fallback simple CNN for single frame classification
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model