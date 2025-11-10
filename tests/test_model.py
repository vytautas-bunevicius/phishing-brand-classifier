"""
Unit tests for phishing classifier models.
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from phishing_classifier.models import BrandClassifier, create_model
from phishing_classifier.models.losses import FocalLoss, get_loss_function


class TestBrandClassifier:
    """Tests for BrandClassifier model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = BrandClassifier(
            backbone="resnet18",  # Use smaller model for testing
            num_classes=11,
            pretrained=False
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = BrandClassifier(
            backbone="resnet18",
            num_classes=11,
            pretrained=False
        )
        model.eval()

        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        assert output.shape == (batch_size, 11)

    def test_feature_extraction(self):
        """Test feature extraction."""
        model = BrandClassifier(
            backbone="resnet18",
            num_classes=11,
            pretrained=False
        )
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(2, 3, 224, 224)

        # Extract features
        with torch.no_grad():
            features = model.extract_features(dummy_input)

        # Check features shape
        assert len(features.shape) == 2
        assert features.shape[0] == 2


class TestLossFunctions:
    """Tests for loss functions."""

    def test_focal_loss(self):
        """Test Focal Loss."""
        loss_fn = FocalLoss(alpha=None, gamma=2.0)

        # Create dummy predictions and targets
        predictions = torch.randn(4, 11)
        targets = torch.tensor([0, 5, 10, 3])

        # Calculate loss
        loss = loss_fn(predictions, targets)

        # Check loss is scalar
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_get_loss_function(self):
        """Test loss function factory."""
        # Test different loss types
        loss_types = ['ce', 'focal', 'label_smoothing']

        for loss_type in loss_types:
            loss_fn = get_loss_function(loss_type)
            assert loss_fn is not None
            assert isinstance(loss_fn, torch.nn.Module)


def test_model_parameters():
    """Test model has trainable parameters."""
    model = BrandClassifier(
        backbone="resnet18",
        num_classes=11,
        pretrained=False
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params == total_params  # All params should be trainable


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
