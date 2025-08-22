# ASL Sign Language Interpreter

Real-time American Sign Language gesture recognition using CNN and GRU neural networks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the ASL Alphabet dataset from Kaggle:
   - https://www.kaggle.com/datasets/grassknoted/asl-alphabet
   - Extract to `asl_alphabet_train/` folder

## Usage

### Training the Model
```bash
python train.py
```
Update the `data_path` variable in `train.py` to point to your dataset location.

### Live Camera Recognition
```bash
python live_camera.py
```

## Project Structure

- `data_preprocessor.py` - Handles dataset loading and preprocessing
- `model.py` - CNN-GRU model architecture
- `train.py` - Training script
- `live_camera.py` - Real-time camera interface
- `requirements.txt` - Dependencies

## Model Architecture

- **CNN**: Extracts spatial features from hand gestures
- **GRU**: Models temporal sequences for improved accuracy
- **Fallback**: Simple CNN for single-frame classification

## Controls

- Place your hand in the green rectangle
- Press 'q' to quit the camera interface

## Dataset

Uses the ASL Alphabet dataset containing 29 classes (A-Z + 3 additional gestures).