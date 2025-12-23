# Phishing Brand Classifier

> **Deep learning-based phishing website detection through screenshot analysis and brand classification**

A production-ready computer vision system that identifies phishing websites by classifying website screenshots into targeted brands, with a strong emphasis on minimizing false positives to avoid flagging legitimate websites.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Data Exploration](#data-exploration)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [API Service](#api-service)
- [Model Architecture](#model-architecture)
- [False Positive Reduction](#false-positive-reduction)
- [Results](#results)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Phishing attacks targeting well-known brands are increasing at an alarming rate. This project addresses the challenge by:

1. **Brand Classification**: Identifying which brand (Amazon, Apple, Facebook, Google, etc.) a website screenshot belongs to
2. **Phishing Detection**: Detecting when a newly registered domain mimics a legitimate brand
3. **False Positive Minimization**: Ensuring benign websites (labeled as "others") are not misclassified as targeted brands

### The Problem

Traditional signature-based phishing detection struggles to keep pace with modern threats. This system uses deep learning to:
- Analyze visual similarities between phishing sites and legitimate brands
- Detect subtle branding cues (logos, colors, layouts)
- Minimize false positives that create poor user experience

### The Solution

A transfer learning approach using state-of-the-art CNN architectures (EfficientNet, ResNet) fine-tuned on website screenshots, with:
- **Focal Loss** for handling class imbalance
- **Custom evaluation metrics** focused on false positive rate
- **Grad-CAM visualizations** for model interpretability
- **FastAPI service** for production deployment

---

## Key Features

### Core Functionality
- **Multi-Brand Classification**: Supports 10 targeted brands + "others" category
- **Transfer Learning**: Pre-trained on ImageNet, fine-tuned on phishing data
- **Class Imbalance Handling**: Focal Loss, class weights, stratified sampling
- **False Positive Reduction**: Custom loss penalties, threshold tuning

### Model Interpretability
- **Grad-CAM Visualizations**: See which regions influenced predictions
- **Comprehensive Metrics**: Precision, recall, F1, ROC-AUC, confusion matrices
- **Error Analysis**: Detailed breakdown of misclassification patterns

### Production Ready
- **REST API**: FastAPI service with batch prediction support
- **Fast Inference**: < 100ms per image on GPU
- **Containerizable**: Easy Docker deployment
- **Comprehensive Testing**: Unit tests and integration tests

### Developer Experience
- **Clean Architecture**: Modular, reusable components
- **Type Hints**: Full type annotations for better IDE support
- **Documentation**: Detailed docstrings and examples
- **Reproducibility**: Fixed seeds, deterministic training

---

## Project Structure

```
phishing-brand-classifier/
├── configs/
│   └── config.yaml           # Training configuration
├── data/
│   ├── raw/                  # Original dataset (brand folders with images)
│   └── processed/            # Processed splits (train/val/test CSVs)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── api/
│   │   └── app.py            # FastAPI application
│   ├── data/
│   │   ├── dataset.py        # PyTorch dataset classes
│   │   ├── transforms.py     # Image augmentation transforms
│   │   └── utils.py          # Data utilities
│   ├── models/
│   │   ├── classifier.py     # Model architectures
│   │   └── losses.py         # Custom loss functions
│   ├── utils/
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualization.py  # Plotting utilities
│   ├── interpretability.py   # GradCAM and model explanations
│   ├── train.py              # Training script
│   └── predict.py            # Inference script
├── outputs/
│   ├── models/               # Saved model checkpoints
│   ├── logs/                 # Training logs
│   └── figures/              # Generated visualizations
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration and dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup with uv (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/vytautas-bunevicius/phishing-brand-classifier.git
cd phishing-brand-classifier
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Download the dataset:
```bash
aws s3 cp s3://phishing-detection-homework-public-bucket data/raw --recursive --no-sign-request
```

### Alternative Setup with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

---

## Dataset

### Download Instructions

The dataset contains website screenshots organized by brand:

```bash
# Option 1: Direct AWS CLI
aws s3 cp s3://phishing-detection-homework-public-bucket data/raw --recursive --no-sign-request
```

### Dataset Structure

```
data/raw/
├── amazon/
│   ├── amazon-login.com.png
│   ├── amaz0n-secure.com.png
│   └── ...
├── apple/
│   ├── apple-id.com.png
│   └── ...
├── facebook/
├── google/
├── instagram/
├── linkedin/
├── microsoft/
├── netflix/
├── paypal/
├── twitter/
└── others/                # Benign websites (not phishing)
```

---

## Usage

### 1. Exploratory Data Analysis

Run the EDA notebook to understand the dataset:
```bash
uv run jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 2. Feature Engineering

Prepare the data for training:
```bash
uv run jupyter notebook notebooks/02_feature_engineering.ipynb
```

This will create train/validation/test splits in `data/processed/`.

### 3. Model Training

#### Using the notebook:
```bash
uv run jupyter notebook notebooks/03_model_training.ipynb
```

#### Using the command line:
```bash
uv run python -m src.train \
    --config configs/config.yaml \
    --data-dir data/raw \
    --output-dir outputs \
    --experiment-name my_experiment
```

### 4. Inference

#### Single image prediction:
```bash
uv run python -m src.predict \
    path/to/screenshot.png \
    --checkpoint outputs/models/best_model.pt \
    --threshold 0.85
```

#### With top-k predictions:
```bash
uv run python -m src.predict \
    path/to/screenshot.png \
    --checkpoint outputs/models/best_model.pt \
    --top-k 3
```

#### Benchmark inference speed:
```bash
uv run python -m src.predict \
    path/to/screenshot.png \
    --checkpoint outputs/models/best_model.pt \
    --benchmark
```

### 5. API Server

Start the REST API:
```bash
# Set environment variables
export MODEL_CHECKPOINT=outputs/models/best_model.pt
export CONFIDENCE_THRESHOLD=0.85

# Run the server
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## Model Architecture

### Transfer Learning Approach

```
Input Image (224x224x3)
        ↓
Pre-trained Backbone (EfficientNet-B0)
        ↓
Global Average Pooling
        ↓
Dropout (0.3)
        ↓
Dense Layer (512 units) + BatchNorm + ReLU
        ↓
Dropout (0.3)
        ↓
Output Layer (11 classes)
        ↓
Softmax Probabilities
```

### Available Architectures

- `efficientnet_b0` (default) - Best accuracy/speed trade-off
- `resnet50` - Good baseline
- `resnet18` - Faster ResNet
- `efficientnet_b3` - Higher accuracy, slower
- `vit_base_patch16_224` - Vision Transformer

---

## False Positive Reduction

### Why It Matters

Falsely flagging legitimate websites as phishing creates:
- Poor user experience
- Loss of trust
- Reduced adoption

### Our Approach

#### 1. Confidence Thresholding
Predictions below a confidence threshold are rejected:
```python
if confidence < threshold:
    prediction = "others"  # Treat as benign
```

#### 2. Focal Loss
Focuses training on hard-to-classify examples:
```python
FocalLoss(gamma=2.0, alpha=class_weights)
```

#### 3. Class Weights
Balances the loss for imbalanced classes.

#### 4. Threshold Optimization
Find optimal confidence threshold to achieve target FPR:
```python
target_fpr = 0.01  # 1% false positive rate
optimal_threshold = find_threshold(y_true, y_pred, target_fpr)
```

### Recommended Threshold

Through threshold optimization (maximizing F1 while keeping false positive rate < 5%), we recommend:
- **Default threshold**: 0.85

---

## Results

### Expected Performance

| Metric | Target |
|--------|--------|
| Test Accuracy | > 90% |
| F1 Score | > 0.88 |
| FP Rate (others) | < 5% |
| Inference (GPU) | < 10 ms |
| Inference (CPU) | < 50 ms |

---

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Classify single image |
| `/predict/batch` | POST | Classify multiple images |
| `/predict/top-k` | POST | Get top-k predictions |
| `/benchmark` | GET | Run inference benchmark |

### Example Usage

```python
import requests

# Single prediction
with open("screenshot.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
print(response.json())
```

### Response Format

```json
{
  "predicted_brand": "amazon",
  "confidence": 0.987,
  "is_phishing": true,
  "all_predictions": {
    "amazon": 0.987,
    "google": 0.008,
    "others": 0.003
  },
  "inference_time_ms": 42.5
}
```

---

## Technical Details

### Data Augmentation Strategy

```python
Training:
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=0.1, scale=0.15, rotate=10°)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
- GaussianBlur / MotionBlur
- GaussianNoise
- ImageCompression (quality 75-100)

Validation/Test:
- Resize to 224x224
- Normalize (ImageNet stats)
```

### Training Configuration

```python
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Scheduler: CosineAnnealing
- Batch Size: 32
- Epochs: 50 (with early stopping)
- Train/Val/Test Split: 70/15/15 (stratified)
```

---

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/
ruff check src/ --fix
```

### Type Checking
```bash
mypy src/
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

---

## Acknowledgments

Built for Nord Security's phishing detection challenge.

---

## Contact

**Vytautas Bunevicius**
- GitHub: [@vytautas-bunevicius](https://github.com/vytautas-bunevicius)
