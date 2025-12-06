"""Brand classifier model architectures."""

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrandClassifier(nn.Module):
    """Brand classifier for phishing detection.

    Uses a pretrained backbone with a custom classification head.
    Supports multiple architectures through the timm library.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 11,
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: Optional[int] = None,
    ):
        """Initialize the classifier.

        Args:
            architecture: Model architecture name (timm compatible).
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate for the classifier head.
            embedding_dim: Optional intermediate embedding dimension.
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Create backbone using timm
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features

        # Custom classification head
        if embedding_dim is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(embedding_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, num_classes),
            )

        # Store the target layer for GradCAM
        self._target_layer = None
        self._set_target_layer()

    def _set_target_layer(self):
        """Set the target layer for GradCAM visualization."""
        if "efficientnet" in self.architecture:
            # For EfficientNet, use the last conv layer
            self._target_layer = self.backbone.conv_head
        elif "resnet" in self.architecture:
            self._target_layer = self.backbone.layer4[-1]
        elif "vit" in self.architecture:
            self._target_layer = self.backbone.blocks[-1].norm1
        else:
            # Try to find the last conv layer
            for name, module in reversed(list(self.backbone.named_modules())):
                if isinstance(module, nn.Conv2d):
                    self._target_layer = module
                    break

    @property
    def target_layer(self):
        """Get the target layer for GradCAM."""
        return self._target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature embeddings of shape (B, feature_dim).
        """
        return self.backbone(x)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        models: List[BrandClassifier],
        weights: Optional[List[float]] = None,
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
    architecture: str = "efficientnet_b0",
    num_classes: int = 11,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
) -> BrandClassifier:
    """Create and optionally load a brand classifier.

    Args:
        architecture: Model architecture name.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        dropout: Dropout rate.
        checkpoint_path: Optional path to load model weights.
        device: Device to load model on.

    Returns:
        Initialized BrandClassifier.
    """
    model = BrandClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    if checkpoint_path is not None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print(f"Loaded model from {checkpoint_path}")

    return model


# Architecture options with their characteristics
ARCHITECTURE_INFO = {
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
    "resnet50": {
        "params": "25.6M",
        "flops": "4.1B",
        "description": "Classic architecture, well understood",
    },
    "resnet18": {
        "params": "11.7M",
        "flops": "1.8B",
        "description": "Fast inference, good for production",
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
