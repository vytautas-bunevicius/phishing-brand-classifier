"""
Brand classifier model architectures.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class BrandClassifier(nn.Module):
    """
    Brand classifier using transfer learning.

    Supports multiple backbone architectures:
    - ResNet variants (resnet50, resnet101)
    - EfficientNet variants (efficientnet_b0 to efficientnet_b7)
    - Vision Transformers (vit_base_patch16_224, vit_large_patch16_224)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 11,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize classifier.

        Args:
            backbone: Name of the backbone architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        # Create backbone model
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=""  # Remove global pooling
        )

        # Get number of features from backbone
        self.num_features = self.model.num_features

        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, 3, height, width]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features
        features = self.model(x)

        # Global pooling
        features = self.global_pool(features)

        # Classification
        logits = self.classifier(features)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Args:
            x: Input tensor [batch_size, 3, height, width]

        Returns:
            Features [batch_size, num_features]
        """
        features = self.model(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        return features


def create_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 11,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> BrandClassifier:
    """
    Create and initialize model.

    Args:
        backbone: Name of the backbone architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        checkpoint_path: Path to load checkpoint from
        device: Device to load model on

    Returns:
        Initialized model
    """
    model = BrandClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {backbone}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model
