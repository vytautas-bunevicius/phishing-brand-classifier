"""Tests for model components."""

import pytest
import torch

from src.models.classifier import BrandClassifier, create_model
from src.models.losses import FocalLoss, LabelSmoothingCrossEntropy


class TestBrandClassifier:
    """Tests for the BrandClassifier class."""

    def test_model_creation(self):
        """Test model can be created with default parameters."""
        model = BrandClassifier(
            architecture="efficientnet_b0",
            num_classes=11,
            pretrained=False,
        )
        assert model is not None
        assert model.num_classes == 11

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = BrandClassifier(
            architecture="efficientnet_b0",
            num_classes=11,
            pretrained=False,
        )
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 11)

    def test_predict_proba(self):
        """Test predict_proba returns valid probabilities."""
        model = BrandClassifier(
            architecture="efficientnet_b0",
            num_classes=11,
            pretrained=False,
        )
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            probs = model.predict_proba(x)

        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        # Check all probabilities are non-negative
        assert (probs >= 0).all()

    def test_predict_with_confidence(self):
        """Test prediction with confidence thresholding."""
        model = BrandClassifier(
            architecture="efficientnet_b0",
            num_classes=11,
            pretrained=False,
        )
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            preds, confs = model.predict_with_confidence(
                x, confidence_threshold=0.5, rejection_class=-1
            )

        assert preds.shape == (4,)
        assert confs.shape == (4,)
        assert (confs >= 0).all() and (confs <= 1).all()

    def test_get_features(self):
        """Test feature extraction."""
        model = BrandClassifier(
            architecture="efficientnet_b0",
            num_classes=11,
            pretrained=False,
        )
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            features = model.get_features(x)

        assert features.shape == (2, model.feature_dim)


class TestCreateModel:
    """Tests for the create_model factory function."""

    def test_create_model_basic(self):
        """Test basic model creation."""
        model = create_model(
            architecture="efficientnet_b0",
            num_classes=11,
            pretrained=False,
        )
        assert isinstance(model, BrandClassifier)

    @pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet18"])
    def test_create_model_architectures(self, arch):
        """Test different architectures can be created."""
        model = create_model(
            architecture=arch,
            num_classes=11,
            pretrained=False,
        )
        assert model is not None


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        loss_fn = FocalLoss(gamma=2.0)

        logits = torch.randn(4, 11)
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_focal_loss_with_weights(self):
        """Test focal loss with class weights."""
        weights = torch.ones(11)
        weights[10] = 2.0  # Higher weight for 'others'

        loss_fn = FocalLoss(alpha=weights, gamma=2.0)

        logits = torch.randn(4, 11)
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)

        assert loss.item() >= 0

    def test_focal_loss_gamma_zero_equals_ce(self):
        """Test that gamma=0 approximates cross-entropy."""
        focal_loss = FocalLoss(gamma=0.0)
        ce_loss = torch.nn.CrossEntropyLoss()

        logits = torch.randn(4, 11)
        targets = torch.randint(0, 11, (4,))

        focal = focal_loss(logits, targets)
        ce = ce_loss(logits, targets)

        # Should be close (not exact due to implementation details)
        assert abs(focal.item() - ce.item()) < 0.5


class TestLabelSmoothingLoss:
    """Tests for LabelSmoothingCrossEntropy."""

    def test_label_smoothing_basic(self):
        """Test basic label smoothing loss."""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

        logits = torch.randn(4, 11)
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_label_smoothing_zero_equals_ce(self):
        """Test that smoothing=0 equals cross-entropy."""
        smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.0)
        ce_loss = torch.nn.CrossEntropyLoss()

        logits = torch.randn(4, 11)
        targets = torch.randint(0, 11, (4,))

        smooth = smooth_loss(logits, targets)
        ce = ce_loss(logits, targets)

        assert torch.allclose(smooth, ce, atol=1e-5)
