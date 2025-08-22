import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessor import ASLDataProcessor
from model import ASLModel
import pickle

def train_model(data_path):
    # Load and preprocess data
    processor = ASLDataProcessor(data_path)
    images, labels = processor.load_data()
    
    print(f"Loaded {len(images)} images with {len(processor.get_classes())} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create and train model
    model_builder = ASLModel(
        input_shape=(64, 64, 3), 
        num_classes=len(processor.get_classes())
    )
    
    # Use simple CNN for this dataset (single frame classification)
    model = model_builder.build_simple_cnn()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,  # Reduced for sample data
        batch_size=16,
        verbose=1
    )
    
    # Save model and label encoder
    model.save('asl_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(processor.label_encoder, f)
    
    print(f"Model saved! Test accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, processor.label_encoder

if __name__ == "__main__":
    # Update this path to your ASL dataset location
    data_path = "asl_alphabet_train"  # Update with actual path
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at: {data_path}")
        print("Please:")
        print("1. Download ASL dataset from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("2. Extract it to 'asl_alphabet_train/' folder")
        print("3. Or update data_path variable to your dataset location")
        exit(1)
    
    train_model(data_path)