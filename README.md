# Phishing Brand Classifier

A deep learning system for detecting phishing websites through brand classification of screenshot images. This project focuses on minimizing false positives to ensure benign websites are not incorrectly flagged as phishing attempts.

## Overview

Phishing attacks increasingly target well-known brands by creating websites that visually mimic legitimate ones. This classifier analyzes website screenshots to identify the brand being impersonated, enabling detection of potential phishing sites.

### Key Features

- **Multi-brand Classification**: Supports 10 major brands (Amazon, Apple, Facebook, Google, Instagram, LinkedIn, Microsoft, Netflix, PayPal, Twitter) plus "others" for benign sites
- **False Positive Minimization**: Special focus on reducing misclassification of benign websites
- **Confidence Thresholding**: Reject uncertain predictions to improve reliability
- **Model Interpretability**: GradCAM visualizations to understand model decisions
- **Production-Ready API**: FastAPI-based REST API for serving predictions
- **Comprehensive Evaluation**: Detailed metrics, error analysis, and inference benchmarking

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

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup with uv (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/your-username/phishing-brand-classifier.git
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
pip install -e .
```

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

Or using the module:
```bash
uv run python -m src.api.app
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Classify single image |
| `/predict/batch` | POST | Classify multiple images |
| `/predict/top-k` | POST | Get top-k predictions |
| `/benchmark` | GET | Run inference benchmark |

#### Example API Usage

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

## Model Architecture

The default model uses **EfficientNet-B0** as the backbone with a custom classification head:

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Classifier**: Dropout → Linear → 11 classes
- **Input size**: 224×224 pixels

Alternative architectures available:
- `resnet50`: Classic ResNet-50
- `resnet18`: Faster ResNet-18
- `efficientnet_b3`: Larger EfficientNet
- `vit_base_patch16_224`: Vision Transformer

## Handling False Positives

Minimizing false positives (benign sites classified as brands) is critical. Our approach:

### 1. Confidence Thresholding
Predictions below a confidence threshold are rejected and classified as "others":
```python
if confidence < threshold:
    prediction = "others"  # Treat as benign
```

### 2. Focal Loss
Focuses training on hard-to-classify examples:
```python
FocalLoss(gamma=2.0, alpha=class_weights)
```

### 3. Class Weights
Balances the loss for imbalanced classes, giving appropriate attention to the "others" class.

### 4. Weighted Sampling
Oversamples minority classes during training to ensure balanced representation.

### Recommended Threshold

Through threshold optimization (maximizing F1 while keeping false positive rate < 5%), we recommend:
- **Default threshold**: 0.85
- Adjustable based on your tolerance for false positives vs. detection rate

## Performance Metrics

### Key Metrics to Monitor

1. **Overall Accuracy**: Percentage of correct predictions
2. **F1 Score (weighted)**: Harmonic mean of precision and recall
3. **False Positive Rate for 'others'**: Critical metric for user experience
4. **Per-class Precision/Recall**: Identify problematic brands
5. **Inference Latency**: Time per prediction (ms)

### Expected Performance

| Metric | Target |
|--------|--------|
| Test Accuracy | > 90% |
| F1 Score | > 0.88 |
| FP Rate (others) | < 5% |
| Inference (GPU) | < 10 ms |
| Inference (CPU) | < 50 ms |

## Model Interpretability

### GradCAM Visualization

The system provides GradCAM visualizations to understand which image regions influenced the prediction:

```python
from src.interpretability import ModelExplainer

explainer = ModelExplainer(model, class_names)
explanation = explainer.explain("screenshot.png", methods=["gradcam"])
explainer.plot_explanation(explanation)
```

This helps:
- Verify the model focuses on brand-relevant features (logos, layouts)
- Debug misclassifications
- Build trust in the system

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model
model:
  architecture: "efficientnet_b0"
  pretrained: true
  dropout: 0.3
  confidence_threshold: 0.85

# Training
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  use_focal_loss: true

# Data
data:
  image_size: 224
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/
isort src/
```

### Type Checking
```bash
mypy src/
```

## Validation Dataset

During the technical interview, you will receive a validation dataset. To evaluate:

```bash
python -m src.predict \
    data/validation/*.png \
    --checkpoint outputs/models/best_model.pt \
    --threshold 0.85 \
    --output results.json
```

## Future Improvements

1. **Ensemble Models**: Combine multiple architectures for better robustness
2. **Test-Time Augmentation**: Average predictions across augmented versions
3. **Online Learning**: Continuously improve with new phishing examples
4. **Multi-Scale Processing**: Handle various screenshot resolutions
5. **Domain Adaptation**: Fine-tune for specific phishing campaigns

## References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built for Nord Security's phishing detection challenge.
