"""Custom loss functions for handling class imbalance and improving robustness."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard examples.
    This is particularly useful for the phishing detection task where we want to:
    1. Handle imbalance between brands and 'others' class
    2. Focus on hard-to-classify samples near decision boundaries

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
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
            label_smoothing: Label smoothing factor.
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
        # Apply label smoothing to targets
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Compute focal weight
        pt = (probs * smooth_targets).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing.

    Label smoothing prevents overconfidence and improves generalization.
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
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for handling false positives.

    This loss asymmetrically weights false positives vs false negatives,
    which is crucial for phishing detection where we want to minimize
    benign sites being classified as brands (false positives).

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        others_class_idx: Optional[int] = None,
        fp_penalty: float = 2.0,
    ):
        """Initialize asymmetric loss.

        Args:
            gamma_neg: Focusing parameter for negative (incorrect) predictions.
            gamma_pos: Focusing parameter for positive (correct) predictions.
            clip: Probability clipping threshold.
            others_class_idx: Index of the 'others' class for extra penalty.
            fp_penalty: Extra penalty multiplier for misclassifying 'others' as a brand.
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.others_class_idx = others_class_idx
        self.fp_penalty = fp_penalty

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            inputs: Logits of shape (B, C).
            targets: Class indices of shape (B,).

        Returns:
            Loss value.
        """
        probs = torch.sigmoid(inputs)

        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # Positive samples
        pos_mask = targets_one_hot.bool()
        probs_pos = probs[pos_mask]
        loss_pos = -torch.log(probs_pos.clamp(min=1e-8))
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1 - probs_pos) ** self.gamma_pos)

        # Negative samples
        neg_mask = ~pos_mask
        probs_neg = probs[neg_mask]
        probs_neg = (probs_neg - self.clip).clamp(min=0)
        loss_neg = -torch.log((1 - probs_neg).clamp(min=1e-8))
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (probs_neg ** self.gamma_neg)

        # Extra penalty for misclassifying 'others' as a brand
        if self.others_class_idx is not None:
            others_mask = targets == self.others_class_idx
            if others_mask.any():
                # Get probabilities for brand classes when true label is 'others'
                brand_probs = probs[others_mask][:, :self.others_class_idx]
                fp_loss = (brand_probs ** 2).sum(dim=1).mean()
                loss = loss_pos.mean() + loss_neg.mean() + self.fp_penalty * fp_loss
                return loss

        return loss_pos.mean() + loss_neg.mean()


def create_loss_function(
    loss_type: str = "focal",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    others_class_idx: Optional[int] = None,
) -> nn.Module:
    """Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('ce', 'focal', 'label_smoothing', 'asymmetric').
        class_weights: Optional class weights.
        focal_gamma: Gamma parameter for focal loss.
        label_smoothing: Label smoothing factor.
        others_class_idx: Index of 'others' class for asymmetric loss.

    Returns:
        Loss function module.
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights,
        )
    elif loss_type == "asymmetric":
        return AsymmetricLoss(others_class_idx=others_class_idx)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
