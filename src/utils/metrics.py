"""Metrics computation and evaluation utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Optional prediction probabilities.
        class_names: Optional class names for reporting.
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dictionary containing various metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Per-class metrics
    metrics["precision_per_class"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    metrics["recall_per_class"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    metrics["f1_per_class"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    # Classification report
    if class_names is not None:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )

    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average=average
                )
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        normalize: 'true', 'pred', 'all', or None.

    Returns:
        Confusion matrix array.
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def get_false_positive_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    others_class_idx: int,
    class_names: List[str],
) -> Dict:
    """Analyze false positives for the 'others' (benign) class.

    This is critical for phishing detection: we want to minimize
    benign websites being incorrectly classified as target brands.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Prediction probabilities.
        others_class_idx: Index of the 'others' class.
        class_names: List of class names.

    Returns:
        Dictionary with false positive analysis.
    """
    # Find samples where true label is 'others' but predicted as a brand
    others_mask = y_true == others_class_idx
    others_true_count = others_mask.sum()

    # False positives: 'others' classified as any brand
    fp_mask = others_mask & (y_pred != others_class_idx)
    fp_count = fp_mask.sum()

    # False positive rate
    fp_rate = fp_count / others_true_count if others_true_count > 0 else 0

    # Analyze which brands 'others' is most commonly misclassified as
    fp_predictions = y_pred[fp_mask]
    brand_misclassification = {}
    for class_idx, class_name in enumerate(class_names):
        if class_idx != others_class_idx:
            count = (fp_predictions == class_idx).sum()
            brand_misclassification[class_name] = int(count)

    # Confidence analysis for false positives
    fp_confidences = y_proba[fp_mask].max(axis=1) if fp_count > 0 else np.array([])

    analysis = {
        "total_others_samples": int(others_true_count),
        "false_positive_count": int(fp_count),
        "false_positive_rate": float(fp_rate),
        "brand_misclassification_counts": brand_misclassification,
        "fp_confidence_stats": {
            "mean": float(fp_confidences.mean()) if len(fp_confidences) > 0 else 0,
            "std": float(fp_confidences.std()) if len(fp_confidences) > 0 else 0,
            "min": float(fp_confidences.min()) if len(fp_confidences) > 0 else 0,
            "max": float(fp_confidences.max()) if len(fp_confidences) > 0 else 0,
        },
    }

    return analysis


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    others_class_idx: int,
    metric: str = "f1_weighted",
    max_fp_rate: float = 0.05,
) -> Tuple[float, Dict]:
    """Find optimal confidence threshold to minimize false positives.

    The threshold is optimized to:
    1. Keep false positive rate for 'others' below max_fp_rate
    2. Maximize the specified metric

    Args:
        y_true: True labels.
        y_proba: Prediction probabilities.
        others_class_idx: Index of 'others' class.
        metric: Metric to optimize ('f1_weighted', 'accuracy', 'precision').
        max_fp_rate: Maximum acceptable false positive rate for 'others'.

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold).
    """
    thresholds = np.arange(0.5, 0.99, 0.01)
    best_threshold = 0.5
    best_metric_value = 0
    best_metrics = {}

    others_mask = y_true == others_class_idx
    brand_mask = ~others_mask

    for threshold in thresholds:
        # Get predictions at this threshold
        max_proba = y_proba.max(axis=1)
        y_pred_threshold = y_proba.argmax(axis=1)

        # Below threshold -> classify as 'others'
        y_pred_threshold[max_proba < threshold] = others_class_idx

        # Calculate false positive rate for 'others'
        fp_count = ((y_true == others_class_idx) & (y_pred_threshold != others_class_idx)).sum()
        fp_rate = fp_count / others_mask.sum() if others_mask.sum() > 0 else 0

        # Skip if FP rate exceeds maximum
        if fp_rate > max_fp_rate:
            continue

        # Calculate target metric
        if metric == "f1_weighted":
            metric_value = f1_score(y_true, y_pred_threshold, average="weighted", zero_division=0)
        elif metric == "accuracy":
            metric_value = accuracy_score(y_true, y_pred_threshold)
        elif metric == "precision":
            metric_value = precision_score(
                y_true, y_pred_threshold, average="weighted", zero_division=0
            )
        else:
            metric_value = f1_score(y_true, y_pred_threshold, average="weighted", zero_division=0)

        # Also calculate brand accuracy (samples that are actually brands)
        brand_accuracy = accuracy_score(y_true[brand_mask], y_pred_threshold[brand_mask]) if brand_mask.sum() > 0 else 0

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
            best_metrics = {
                "threshold": float(threshold),
                f"{metric}": float(metric_value),
                "false_positive_rate": float(fp_rate),
                "brand_accuracy": float(brand_accuracy),
                "overall_accuracy": float(accuracy_score(y_true, y_pred_threshold)),
            }

    return best_threshold, best_metrics


def evaluate_with_rejection(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    confidence_threshold: float,
    rejection_label: int = -1,
) -> Dict:
    """Evaluate model with rejection option.

    Low-confidence predictions are rejected rather than classified.

    Args:
        y_true: True labels.
        y_proba: Prediction probabilities.
        confidence_threshold: Minimum confidence for prediction.
        rejection_label: Label for rejected samples.

    Returns:
        Evaluation metrics with rejection statistics.
    """
    max_proba = y_proba.max(axis=1)
    y_pred = y_proba.argmax(axis=1)

    # Mark low-confidence predictions as rejected
    rejected_mask = max_proba < confidence_threshold
    y_pred_with_rejection = y_pred.copy()
    y_pred_with_rejection[rejected_mask] = rejection_label

    # Metrics on accepted predictions only
    accepted_mask = ~rejected_mask
    if accepted_mask.sum() > 0:
        accepted_accuracy = accuracy_score(y_true[accepted_mask], y_pred[accepted_mask])
        accepted_f1 = f1_score(
            y_true[accepted_mask], y_pred[accepted_mask], average="weighted", zero_division=0
        )
    else:
        accepted_accuracy = 0
        accepted_f1 = 0

    return {
        "total_samples": len(y_true),
        "accepted_samples": int(accepted_mask.sum()),
        "rejected_samples": int(rejected_mask.sum()),
        "rejection_rate": float(rejected_mask.mean()),
        "accepted_accuracy": float(accepted_accuracy),
        "accepted_f1": float(accepted_f1),
        "confidence_threshold": float(confidence_threshold),
        "avg_confidence_accepted": float(max_proba[accepted_mask].mean()) if accepted_mask.sum() > 0 else 0,
        "avg_confidence_rejected": float(max_proba[rejected_mask].mean()) if rejected_mask.sum() > 0 else 0,
    }


class MetricTracker:
    """Track metrics during training."""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "learning_rate": [],
        }

    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)

    def get_best_epoch(self, metric: str = "val_f1") -> int:
        """Get the epoch with best metric value."""
        if metric not in self.history or not self.history[metric]:
            return 0
        return int(np.argmax(self.history[metric]))

    def get_history(self) -> Dict:
        """Get full history."""
        return self.history
