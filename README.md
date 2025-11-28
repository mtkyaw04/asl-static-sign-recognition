# ASL Static Sign Hand Detection Model

A deep learning model for detecting American Sign Language (ASL) static hand signs using MediaPipe hand landmarks. This model is designed for static signs only (no motion required) and supports digits 0-9 and alphabets A-Z (excluding J and Z, which require motion).

## Overview

This project provides a baseline implementation for ASL sign language recognition that can be integrated into learning applications. The model uses MediaPipe to extract hand landmarks and a PyTorch neural network to classify static ASL signs.

### Supported Signs

- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Alphabets**: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
- **Special**: "I love you" sign

**Note**: J and Z are excluded as they require motion, which this static model cannot detect.

## Project Structure

```
.
├── collect_landmarks.py      # Data collection script
├── train_landmark_model.py   # Model training script
├── predict_static.py         # Real-time prediction script
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── dataset_landmarks/        # Training data (JSON files)
└── README.md                 # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd language
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have a webcam** connected for data collection and prediction.

## Usage

### 1. Data Collection

Use `collect_landmarks.py` to collect hand landmark data for training:

```bash
python collect_landmarks.py
```

**How it works:**
- Enter the label (e.g., 'a', 'b', '1', '2') when prompted
- Press **SPACE** to capture a hand landmark sample
- Press **ESC** to exit and save
- Data is saved as JSON files in `dataset_landmarks/` directory

**Tips:**
- Collect multiple samples per sign (recommended: 50-100+ per class)
- Vary hand positions, angles, and distances from camera
- Ensure good lighting and clear hand visibility

### 2. Model Training

Train the model using your collected data:

```bash
python train_landmark_model.py
```

**What it does:**
- Loads all JSON files from `dataset_landmarks/`
- Splits data into training (80%) and validation (20%) sets
- Trains a neural network with early stopping
- Saves the best model as `best_landmark_model.pth`

**Training Configuration:**
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 100 (with early stopping)
- Patience: 10 epochs
- Optimizer: Adam

**Model Architecture:**
```
Input (42 features: 21 landmarks × 2 coordinates)
    ↓
Linear(42 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(64 → num_classes)
    ↓
Output (class probabilities)
```

### 3. Real-time Prediction

Run predictions on live webcam feed:

```bash
python predict_static.py
```

**How it works:**
- Opens webcam feed
- Detects hand using MediaPipe
- Extracts 21 hand landmarks (x, y coordinates)
- Predicts sign class with confidence score
- Displays prediction overlay on video
- Press **ESC** to exit

## Architecture Details

### Data Format

Each training sample consists of 42 features:
- 21 hand landmarks from MediaPipe
- Each landmark has x and y coordinates (normalized 0-1)
- z coordinates are excluded for static sign recognition

### Model Architecture

The `HandSignClassifier` is a feedforward neural network with:

1. **Input Layer**: 42 features (21 landmarks × 2 coordinates)
2. **Hidden Layer 1**: 128 neurons with BatchNorm, ReLU, and 30% dropout
3. **Hidden Layer 2**: 64 neurons with BatchNorm, ReLU, and 20% dropout
4. **Output Layer**: Number of classes (one per sign)

**Key Features:**
- Batch normalization for stable training
- Dropout for regularization and preventing overfitting
- Early stopping to prevent overtraining
- Cross-entropy loss for multi-class classification

### File Descriptions

#### `collect_landmarks.py`
- **Purpose**: Collect training data by capturing hand landmarks from webcam
- **Input**: User input (label name) and webcam feed
- **Output**: JSON files containing landmark arrays for each sign class
- **Usage**: Run interactively, press SPACE to capture samples

#### `train_landmark_model.py`
- **Purpose**: Train the neural network classifier
- **Input**: JSON files from `dataset_landmarks/`
- **Output**: `best_landmark_model.pth` (trained model weights)
- **Features**: 
  - Automatic train/validation split
  - Early stopping
  - Performance metrics (accuracy, confusion matrix)
  - Model checkpointing

#### `predict_static.py`
- **Purpose**: Real-time sign prediction from webcam
- **Input**: Webcam feed
- **Output**: Live video with predicted sign and confidence
- **Requirements**: Pre-trained `best_landmark_model.pth` file

#### `utils.py`
- **Purpose**: Utility functions for device detection and class name loading
- **Functions**:
  - `get_device()`: Returns CUDA device if available, else CPU
  - `load_class_names()`: Loads class names from directory structure

## Using as a Baseline

This project can serve as a baseline for your ASL learning website or other sign language recognition projects. Here's how to adapt it:

### Integration Steps

1. **Collect Your Data:**
   - Use `collect_landmarks.py` to gather data for your specific use case
   - Adjust the number of samples per class based on your needs
   - Consider adding more signs or modifying existing ones

2. **Train Your Model:**
   - Run `train_landmark_model.py` with your dataset
   - Monitor training metrics to ensure good performance
   - Adjust hyperparameters in the script if needed

3. **Integrate into Your Application:**
   - Import the model architecture from `predict_static.py`
   - Load your trained model weights
   - Use MediaPipe hand detection in your web application
   - Process landmarks and get predictions

### Customization Options

**Add More Signs:**
- Collect data for new signs using `collect_landmarks.py`
- Retrain the model - it will automatically detect new classes

**Modify Model Architecture:**
- Edit `HandSignClassifier` in `train_landmark_model.py`
- Adjust layer sizes, add/remove layers, change activation functions
- Update the model definition in `predict_static.py` to match

**Improve Accuracy:**
- Collect more diverse training data
- Increase model capacity (more neurons/layers)
- Adjust dropout rates and learning rate
- Use data augmentation techniques

**Web Integration:**
- Use MediaPipe JavaScript SDK for browser-based detection
- Send landmarks to a backend API running the PyTorch model
- Or convert model to TensorFlow.js for client-side inference

## Requirements

- Python 3.7+
- OpenCV 4.12.0+
- MediaPipe 0.10.14+
- PyTorch 2.8.0+
- NumPy 2.2.6+
- scikit-learn 1.16.1+
- tqdm 4.66.1+

## Limitations

- **Static signs only**: Cannot detect signs requiring motion (J, Z)
- **Single hand**: Currently supports one hand detection (Left or Right)
- **Webcam required**: Needs camera access for real-time prediction
- **Training data**: Model performance depends on quality and quantity of training data

## Future Improvements

- Support for two-hand signs
- Motion-based sign recognition (for J, Z, and other dynamic signs)
- Data augmentation techniques
- Model quantization for faster inference
- Web-based deployment options
- Mobile app integration

## License
This project is open-source and distributed under the MIT License.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MediaPipe for hand landmark detection
- PyTorch for deep learning framework

