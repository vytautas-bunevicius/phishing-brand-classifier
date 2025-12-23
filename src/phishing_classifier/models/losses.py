"""
Loss functions for handling class imbalance and false positive reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss = -alpha * (1 - p)^gamma * log(p)

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)

        # Get probabilities of target class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        pt = (probs * targets_one_hot).sum(dim=1)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Apply focal term
        loss = focal_term * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss with penalty for false positives on "others" class.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        false_positive_penalty: float = 2.0
    ):
        """
        Initialize Weighted Cross Entropy Loss.

        Args:
            class_weights: Weights for each class
            false_positive_penalty: Additional penalty for misclassifying "others" as brand
        """
        super().__init__()
        self.class_weights = class_weights
        self.false_positive_penalty = false_positive_penalty

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Standard cross entropy
        base_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)

        # Additional penalty for false positives
        # When target is "others" (class 10) but predicted as brand (0-9)
        others_class = inputs.size(1) - 1
        others_mask = (targets == others_class)

        if others_mask.sum() > 0:
            # Get predictions for "others" samples
            others_preds = inputs[others_mask]

            # Probability of predicting any brand instead of others
            brand_probs = F.softmax(others_preds, dim=1)[:, :-1].sum(dim=1)

            # Add penalty
            fp_penalty = (brand_probs * self.false_positive_penalty).mean()
            base_loss = base_loss + fp_penalty

        return base_loss


def get_loss_function(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    false_positive_penalty: float = 2.0
) -> nn.Module:
    """
    Get loss function based on configuration.

    Args:
        loss_type: Type of loss ('ce', 'focal', 'label_smoothing', 'weighted')
        class_weights: Class weights for handling imbalance
        alpha: Alpha parameter for focal loss
        gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor
        false_positive_penalty: Penalty for false positives

    Returns:
        Loss function
    """
    if loss_type == 'focal':
        # Focal loss for hard example mining
        alpha_tensor = None
        if class_weights is not None:
            alpha_tensor = class_weights * alpha
        return FocalLoss(alpha=alpha_tensor, gamma=gamma)

    elif loss_type == 'label_smoothing':
        # Cross entropy with label smoothing
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    elif loss_type == 'weighted':
        # Custom weighted loss with FP penalty
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            false_positive_penalty=false_positive_penalty
        )

    else:  # 'ce' or default
        # Standard cross entropy
        return nn.CrossEntropyLoss(weight=class_weights)
