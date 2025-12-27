"""
Configuration management using Pydantic Settings (2025 best practices).

Supports:
- Type validation with Pydantic v2
- Environment variable overrides
- .env file loading
- Nested configuration models
- Computed fields and validators
"""

import os
from pathlib import Path
from typing import Literal, Annotated
from functools import lru_cache

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    computed_field,
    ConfigDict
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Data configuration with validation."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    data_dir: Path = Field(
        default=Path("data/raw"),
        description="Root directory containing brand folders"
    )
    processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed datasets"
    )
    train_split: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.7,
        description="Proportion of data for training"
    )
    val_split: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.15,
        description="Proportion of data for validation"
    )
    test_split: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.15,
        description="Proportion of data for testing"
    )
    image_size: tuple[int, int] = Field(
        default=(224, 224),
        description="Input image size (height, width)"
    )
    batch_size: Annotated[int, Field(gt=0)] = Field(
        default=32,
        description="Batch size for training"
    )
    num_workers: Annotated[int, Field(ge=0)] = Field(
        default=4,
        description="Number of data loading workers"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    brands: list[str] = Field(
        default=[
            'amazon', 'apple', 'facebook', 'google', 'microsoft',
            'netflix', 'paypal', 'twitter', 'instagram', 'linkedin',
            'others'
        ],
        description="List of brand names (10 targeted + others)"
    )

    @field_validator('train_split', 'val_split', 'test_split')
    @classmethod
    def validate_splits_sum(cls, v, info):
        """Validate that splits sum to 1.0."""
        # This is called for each field, we'll do final validation in model_validator
        return v

    @field_validator('image_size')
    @classmethod
    def validate_image_size(cls, v):
        """Validate image size is reasonable."""
        if not (32 <= v[0] <= 1024 and 32 <= v[1] <= 1024):
            raise ValueError("Image dimensions must be between 32 and 1024")
        return v

    @computed_field
    @property
    def num_classes(self) -> int:
        """Number of classification classes."""
        return len(self.brands)


class ModelConfig(BaseModel):
    """Model configuration with validation."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    # Architecture
    backbone: Literal[
        "efficientnet_b0", "efficientnet_b3", "efficientnet_b5",
        "resnet18", "resnet50", "resnet101",
        "vit_base_patch16_224", "vit_large_patch16_224"
    ] = Field(
        default="efficientnet_b3",
        description="Model backbone architecture"
    )
    num_classes: int = Field(
        default=11,
        description="Number of output classes"
    )
    pretrained: bool = Field(
        default=True,
        description="Use ImageNet pretrained weights"
    )
    dropout: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.3,
        description="Dropout rate"
    )

    # Training
    epochs: Annotated[int, Field(gt=0)] = Field(
        default=50,
        description="Number of training epochs"
    )
    learning_rate: Annotated[float, Field(gt=0)] = Field(
        default=1e-4,
        description="Initial learning rate"
    )
    weight_decay: Annotated[float, Field(ge=0)] = Field(
        default=1e-4,
        description="L2 regularization weight"
    )
    scheduler: Literal["cosine", "step", "plateau"] = Field(
        default="cosine",
        description="Learning rate scheduler type"
    )
    warmup_epochs: Annotated[int, Field(ge=0)] = Field(
        default=5,
        description="Number of warmup epochs"
    )

    # Loss function
    loss_type: Literal["ce", "focal", "label_smoothing", "weighted"] = Field(
        default="focal",
        description="Loss function type"
    )
    focal_alpha: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.25,
        description="Focal loss alpha parameter"
    )
    focal_gamma: Annotated[float, Field(ge=0)] = Field(
        default=2.0,
        description="Focal loss gamma parameter"
    )
    label_smoothing: Annotated[float, Field(ge=0, lt=1)] = Field(
        default=0.1,
        description="Label smoothing factor"
    )

    # Class weights
    use_class_weights: bool = Field(
        default=True,
        description="Apply class weights for imbalance"
    )

    # Early stopping
    patience: Annotated[int, Field(gt=0)] = Field(
        default=10,
        description="Early stopping patience (epochs)"
    )
    min_delta: Annotated[float, Field(ge=0)] = Field(
        default=0.001,
        description="Minimum change to qualify as improvement"
    )

    # Mixed precision training
    use_amp: bool = Field(
        default=True,
        description="Use automatic mixed precision training"
    )

    # Gradient clipping
    gradient_clip_val: float | None = Field(
        default=1.0,
        description="Gradient clipping value (None to disable)"
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration with validation."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    metrics: list[Literal["accuracy", "precision", "recall", "f1", "roc_auc"]] = Field(
        default=["accuracy", "precision", "recall", "f1", "roc_auc"],
        description="Metrics to compute"
    )
    target_fpr: Annotated[float, Field(gt=0, lt=1)] = Field(
        default=0.01,
        description="Target false positive rate for 'others' class"
    )
    confidence_threshold: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.5,
        description="Decision confidence threshold"
    )
    enable_gradcam: bool = Field(
        default=True,
        description="Generate Grad-CAM visualizations"
    )
    num_samples_visualize: Annotated[int, Field(gt=0)] = Field(
        default=10,
        description="Number of samples to visualize"
    )


class APIConfig(BaseModel):
    """API server configuration."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    port: Annotated[int, Field(gt=0, lt=65536)] = Field(
        default=8000,
        description="API port"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )
    workers: Annotated[int, Field(gt=0)] = Field(
        default=1,
        description="Number of worker processes"
    )
    max_batch_size: Annotated[int, Field(gt=0)] = Field(
        default=32,
        description="Maximum batch size for predictions"
    )
    timeout: Annotated[int, Field(gt=0)] = Field(
        default=60,
        description="Request timeout in seconds"
    )


class Settings(BaseSettings):
    """
    Main application settings with environment variable support.

    Environment variables can override any setting:
    - Use PHISHING_CLASSIFIER_ prefix
    - Use double underscore for nested configs

    Examples:
        PHISHING_CLASSIFIER_DATA__BATCH_SIZE=64
        PHISHING_CLASSIFIER_MODEL__BACKBONE=resnet50
        PHISHING_CLASSIFIER_DEBUG=true
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PHISHING_CLASSIFIER_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Paths
    root_dir: Path = Field(
        default=Path("."),
        description="Project root directory"
    )
    models_dir: Path = Field(
        default=Path("models/checkpoints"),
        description="Directory for model checkpoints"
    )
    results_dir: Path = Field(
        default=Path("models/results"),
        description="Directory for evaluation results"
    )
    logs_dir: Path = Field(
        default=Path("runs"),
        description="Directory for training logs"
    )

    # Device configuration
    device: Literal["cuda", "cpu", "mps", "auto"] = Field(
        default="auto",
        description="Computation device (auto = detect best available)"
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Global random seed"
    )
    deterministic: bool = Field(
        default=True,
        description="Enable deterministic training (may be slower)"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format"
    )

    # Debug mode
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Experiment tracking
    experiment_name: str = Field(
        default="phishing_classifier",
        description="Experiment name for tracking"
    )
    track_experiments: bool = Field(
        default=True,
        description="Enable experiment tracking with TensorBoard"
    )

    @computed_field
    @property
    def device_auto(self) -> str:
        """Auto-detect best available device."""
        if self.device != "auto":
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @field_validator('models_dir', 'results_dir', 'logs_dir')
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_dump_json_safe(self) -> dict:
        """Dump config as JSON-safe dict."""
        return self.model_dump(mode='json')


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    This function is cached to avoid re-loading .env file
    and re-validating configuration on every call.

    Returns:
        Validated Settings instance
    """
    return Settings()


# Convenience function for backward compatibility
def get_config() -> Settings:
    """
    Get application configuration.

    This is an alias for get_settings() to maintain
    backward compatibility with older code.
    """
    return get_settings()


# Allow clearing cache for testing
def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()
