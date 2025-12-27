"""Brand classifier model architectures."""

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for model creation using Pydantic v2."""

    backbone: str = Field(default="efficientnet_b3", description="Model architecture")
    num_classes: int = Field(default=11, gt=0, description="Number of output classes")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    dropout: float = Field(default=0.3, ge=0, le=1, description="Dropout rate")
    embedding_dim: int = Field(
        default=512, gt=0, description="Hidden layer dimension in classifier head"
    )


class BrandClassifier(nn.Module):
    """Brand classifier for phishing detection.

    Uses a pretrained backbone with a custom classification head.
    Supports multiple architectures through the timm library.

    Architectures supported:
    - EfficientNet variants (efficientnet_b0 to efficientnet_b7)
    - ResNet variants (resnet18, resnet50, resnet101)
    - Vision Transformers (vit_base_patch16_224, vit_large_patch16_224)
    - ConvNeXt variants (convnext_tiny, convnext_small, convnext_base)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 11,
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: int = 512,
    ):
        """Initialize the classifier.

        Args:
            backbone: Model architecture name (timm compatible).
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate for the classifier head.
            embedding_dim: Intermediate embedding dimension.
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Create backbone using timm
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            global_pool="",  # Remove global pooling
        )

        # Get feature dimension from backbone
        self.num_features = self.model.num_features

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Custom classification head with hidden layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

        # Store the target layer for GradCAM
        self._target_layer: Optional[nn.Module] = None
        self._set_target_layer()

    def _set_target_layer(self) -> None:
        """Set the target layer for GradCAM visualization."""
        if "efficientnet" in self.backbone_name:
            self._target_layer = self.model.conv_head
        elif "resnet" in self.backbone_name:
            self._target_layer = self.model.layer4[-1]
        elif "vit" in self.backbone_name:
            self._target_layer = self.model.blocks[-1].norm1
        elif "convnext" in self.backbone_name:
            self._target_layer = self.model.stages[-1]
        else:
            # Try to find the last conv layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    self._target_layer = module
                    break

    @property
    def target_layer(self) -> Optional[nn.Module]:
        """Get the target layer for GradCAM."""
        return self._target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.model(x)
        features = self.global_pool(features)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before classification.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature embeddings of shape (B, num_features).
        """
        features = self.model(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        return features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Probabilities of shape (B, num_classes).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.85,
        rejection_class: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with confidence thresholding.

        Predictions below the confidence threshold are rejected
        to minimize false positives (benign sites classified as brands).

        Args:
            x: Input tensor of shape (B, C, H, W).
            confidence_threshold: Minimum confidence for prediction.
            rejection_class: Class index for rejected predictions.

        Returns:
            Tuple of (predictions, confidences).
        """
        probs = self.predict_proba(x)
        confidences, predictions = probs.max(dim=1)

        # Reject low-confidence predictions
        mask = confidences < confidence_threshold
        predictions[mask] = rejection_class

        return predictions, confidences


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple classifiers for improved robustness."""

    def __init__(
        self,
        models: list[BrandClassifier],
        weights: Optional[list[float]] = None,
    ):
        """Initialize the ensemble.

        Args:
            models: List of trained classifiers.
            weights: Optional weights for each model (should sum to 1).
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = models[0].num_classes

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted averaging.

        Args:
            x: Input tensor.

        Returns:
            Averaged logits.
        """
        outputs = []
        for model, weight in zip(self.models, self.weights):
            outputs.append(model(x) * weight)
        return torch.stack(outputs).sum(dim=0)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get averaged probabilities."""
        probs = []
        for model, weight in zip(self.models, self.weights):
            probs.append(model.predict_proba(x) * weight)
        return torch.stack(probs).sum(dim=0)


def create_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 11,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    embedding_dim: int = 512,
) -> BrandClassifier:
    """Create and optionally load a brand classifier.

    Args:
        backbone: Model architecture name.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        dropout: Dropout rate.
        checkpoint_path: Optional path to load model weights.
        device: Device to load model on ('auto', 'cuda', 'cpu', 'mps').
        embedding_dim: Hidden layer dimension in classifier head.

    Returns:
        Initialized BrandClassifier.
    """
    model = BrandClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        embedding_dim=embedding_dim,
    )

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        logger.info(f"Loaded model from {checkpoint_path}")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {backbone}, Total params: {total_params:,}, "
        f"Trainable: {trainable_params:,}, Device: {device}"
    )

    return model


# Architecture options with their characteristics
ARCHITECTURE_INFO: dict[str, dict[str, str]] = {
    "efficientnet_b0": {
        "params": "5.3M",
        "flops": "0.39B",
        "description": "Good balance of speed and accuracy",
    },
    "efficientnet_b3": {
        "params": "12M",
        "flops": "1.8B",
        "description": "Better accuracy, moderate speed",
    },
    "efficientnet_b5": {
        "params": "30M",
        "flops": "9.9B",
        "description": "High accuracy, slower inference",
    },
    "resnet18": {
        "params": "11.7M",
        "flops": "1.8B",
        "description": "Fast inference, good for production",
    },
    "resnet50": {
        "params": "25.6M",
        "flops": "4.1B",
        "description": "Classic architecture, well understood",
    },
    "vit_base_patch16_224": {
        "params": "86M",
        "flops": "17.6B",
        "description": "Vision Transformer, excellent for complex patterns",
    },
    "convnext_tiny": {
        "params": "28.6M",
        "flops": "4.5B",
        "description": "Modern CNN, competitive with ViT",
    },
}
