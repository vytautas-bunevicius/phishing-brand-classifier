"""Loss functions for handling class imbalance and false positive reduction."""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    """Configuration for loss functions using Pydantic v2."""

    loss_type: Literal["ce", "focal", "label_smoothing", "weighted"] = Field(
        default="focal", description="Type of loss function"
    )
    gamma: float = Field(default=2.0, ge=0, description="Focal loss gamma parameter")
    alpha: float = Field(default=0.25, gt=0, lt=1, description="Focal loss alpha")
    label_smoothing: float = Field(
        default=0.1, ge=0, lt=1, description="Label smoothing factor"
    )
    false_positive_penalty: float = Field(
        default=2.0, ge=0, description="Penalty for false positives on 'others' class"
    )


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal Loss = -alpha * (1 - p)^gamma * log(p)

    This loss down-weights well-classified examples and focuses on hard examples.
    Particularly useful for the phishing detection task where we want to:
    1. Handle imbalance between brands and 'others' class
    2. Focus on hard-to-classify samples near decision boundaries

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Optional class weights tensor of shape (num_classes,).
                   Higher values for minority classes.
            gamma: Focusing parameter. Higher values = more focus on hard examples.
                   gamma=0 is equivalent to cross-entropy.
            reduction: 'none', 'mean', or 'sum'.
            label_smoothing: Label smoothing factor (0 = no smoothing).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (B, C).
            targets: Class indices of shape (B,).

        Returns:
            Loss value.
        """
        num_classes = inputs.size(1)

        # Get probabilities
        probs = F.softmax(inputs, dim=1)

        # Get probabilities of target class
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        pt = (probs * targets_one_hot).sum(dim=1)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Apply focal term
        loss = focal_term * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing.

    Label smoothing prevents overconfidence and improves generalization
    by softening the target distribution.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """Initialize label smoothing loss.

        Args:
            smoothing: Smoothing factor (0 = no smoothing).
            weight: Optional class weights.
            reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross entropy.

        Args:
            inputs: Logits of shape (B, C).
            targets: Class indices of shape (B,).

        Returns:
            Loss value.
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)

        # Create smooth labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=1)

        # Apply class weights
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            loss = loss * weight[targets]

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss with penalty for false positives on 'others' class.

    This is crucial for phishing detection where we want to minimize
    benign sites being classified as brands (false positives).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        false_positive_penalty: float = 2.0,
    ):
        """Initialize Weighted Cross Entropy Loss.

        Args:
            class_weights: Weights for each class.
            false_positive_penalty: Additional penalty for misclassifying 'others' as brand.
        """
        super().__init__()
        self.class_weights = class_weights
        self.false_positive_penalty = false_positive_penalty

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy with false positive penalty.

        Args:
            inputs: Logits of shape (B, C).
            targets: Class indices of shape (B,).

        Returns:
            Loss value.
        """
        weights = self.class_weights.to(inputs.device) if self.class_weights is not None else None
        base_loss = F.cross_entropy(inputs, targets, weight=weights)

        # Additional penalty for false positives
        # When target is "others" (last class) but predicted as brand
        others_class = inputs.size(1) - 1
        others_mask = targets == others_class

        if others_mask.sum() > 0:
            others_preds = inputs[others_mask]
            # Probability of predicting any brand instead of others
            brand_probs = F.softmax(others_preds, dim=1)[:, :-1].sum(dim=1)
            fp_penalty = (brand_probs * self.false_positive_penalty).mean()
            base_loss = base_loss + fp_penalty

        return base_loss


def get_loss_function(
    loss_type: str = "focal",
    class_weights: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    false_positive_penalty: float = 2.0,
) -> nn.Module:
    """Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('ce', 'focal', 'label_smoothing', 'weighted').
        class_weights: Optional class weights for handling imbalance.
        alpha: Alpha parameter for focal loss.
        gamma: Gamma parameter for focal loss.
        label_smoothing: Label smoothing factor.
        false_positive_penalty: Penalty for false positives on 'others' class.

    Returns:
        Loss function module.

    Raises:
        ValueError: If unknown loss type is specified.
    """
    loss_type = loss_type.lower()

    if loss_type == "focal":
        alpha_tensor = None
        if class_weights is not None:
            alpha_tensor = class_weights * alpha
        return FocalLoss(
            alpha=alpha_tensor,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights,
        )

    elif loss_type == "weighted":
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            false_positive_penalty=false_positive_penalty,
        )

    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: 'ce', 'focal', 'label_smoothing', 'weighted'"
        )
