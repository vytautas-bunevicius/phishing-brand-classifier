"""
Configuration management for the phishing classifier.
"""

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

    # Brand names (10 targeted brands + others)
    brands: List[str] = field(default_factory=lambda: [
        'amazon', 'apple', 'facebook', 'google', 'microsoft',
        'netflix', 'paypal', 'twitter', 'instagram', 'linkedin',
        'others'
    ])


@dataclass
class ModelConfig:
    """Model configuration."""

    # Architecture
    backbone: str = "efficientnet_b3"  # Options: resnet50, efficientnet_b3, vit_base_patch16_224
    num_classes: int = 11  # 10 brands + others
    pretrained: bool = True
    dropout: float = 0.3

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    warmup_epochs: int = 5

    # Loss function
    loss_type: str = "focal"  # Options: ce, focal, label_smoothing
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # Class weights for handling imbalance
    use_class_weights: bool = True

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    ])

    # False positive reduction
    target_fpr: float = 0.01  # Target false positive rate for "others" class
    confidence_threshold: float = 0.5  # Decision threshold

    # Interpretability
    enable_gradcam: bool = True
    num_samples_visualize: int = 10


@dataclass
class Config:
    """Main configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Paths
    root_dir: Path = Path(".")
    models_dir: Path = Path("models/checkpoints")
    results_dir: Path = Path("models/results")
    logs_dir: Path = Path("runs")

    # Device
    device: str = "cuda"  # Will be auto-detected

    # Reproducibility
    seed: int = 42
    deterministic: bool = True


def get_config() -> Config:
    """Get default configuration."""
    return Config()
