"""
Evaluation metrics with focus on false positive reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import torch


class MetricsCalculator:
    """Calculate and track evaluation metrics."""

    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate comprehensive metrics.

        Args:
            y_true: True labels [num_samples]
            y_pred: Predicted labels [num_samples]
            y_proba: Prediction probabilities [num_samples, num_classes]

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Weighted metrics (account for class imbalance)
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics['per_class_precision'] = {
            name: float(score)
            for name, score in zip(self.class_names, per_class_precision)
        }
        metrics['per_class_recall'] = {
            name: float(score)
            for name, score in zip(self.class_names, per_class_recall)
        }
        metrics['per_class_f1'] = {
            name: float(score)
            for name, score in zip(self.class_names, per_class_f1)
        }

        # ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                metrics['macro_roc_auc'] = roc_auc_score(
                    y_true, y_proba, average='macro', multi_class='ovr'
                )
                metrics['weighted_roc_auc'] = roc_auc_score(
                    y_true, y_proba, average='weighted', multi_class='ovr'
                )
            except ValueError:
                # Handle cases where some classes might not be present
                metrics['macro_roc_auc'] = 0.0
                metrics['weighted_roc_auc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # False positive analysis for "others" class
        others_idx = self.num_classes - 1  # Assuming "others" is last class
        others_metrics = self._analyze_others_class(cm, others_idx)
        metrics.update(others_metrics)

        return metrics

    def _analyze_others_class(
        self,
        confusion_matrix: np.ndarray,
        others_idx: int
    ) -> Dict:
        """
        Analyze false positives specifically for the "others" class.

        This is critical because we want to minimize benign websites
        being classified as phishing (targeted brands).

        Args:
            confusion_matrix: Confusion matrix
            others_idx: Index of "others" class

        Returns:
            Dictionary with others-specific metrics
        """
        metrics = {}

        # Total "others" samples
        total_others = confusion_matrix[others_idx, :].sum()

        if total_others > 0:
            # Correctly classified as "others"
            true_negatives = confusion_matrix[others_idx, others_idx]

            # Misclassified as brands (false positives)
            false_positives = total_others - true_negatives

            # False positive rate for "others"
            fpr_others = false_positives / total_others

            metrics['others_total'] = int(total_others)
            metrics['others_correct'] = int(true_negatives)
            metrics['others_false_positives'] = int(false_positives)
            metrics['others_fpr'] = float(fpr_others)
            metrics['others_accuracy'] = float(true_negatives / total_others)

            # Which brands are "others" most commonly misclassified as?
            others_row = confusion_matrix[others_idx, :]
            misclassified_as = {}
            for i, count in enumerate(others_row):
                if i != others_idx and count > 0:
                    misclassified_as[self.class_names[i]] = int(count)

            metrics['others_misclassified_as'] = misclassified_as

        # Brands misclassified as "others" (false negatives for brands)
        brand_false_negatives = {}
        for i in range(len(self.class_names)):
            if i != others_idx:
                total_brand = confusion_matrix[i, :].sum()
                if total_brand > 0:
                    fn = confusion_matrix[i, others_idx]
                    brand_false_negatives[self.class_names[i]] = {
                        'count': int(fn),
                        'rate': float(fn / total_brand)
                    }

        metrics['brand_false_negatives'] = brand_false_negatives

        return metrics

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )


def calculate_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    others_class_idx: int,
    target_fpr: float = 0.01
) -> Tuple[float, Dict]:
    """
    Calculate optimal classification threshold to achieve target FPR for "others" class.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities [num_samples, num_classes]
        others_class_idx: Index of "others" class
        target_fpr: Target false positive rate

    Returns:
        optimal_threshold: Threshold value
        metrics: Dictionary of metrics at this threshold
    """
    # Create binary labels: others vs brands
    y_true_binary = (y_true == others_class_idx).astype(int)

    # Probability of "others" class
    y_proba_others = y_proba[:, others_class_idx]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba_others)

    # Find threshold closest to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    optimal_threshold = thresholds[idx]

    metrics = {
        'threshold': float(optimal_threshold),
        'fpr': float(fpr[idx]),
        'tpr': float(tpr[idx]),
        'target_fpr': target_fpr
    }

    return optimal_threshold, metrics


def calculate_inference_time(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int],
    device: str,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Calculate model inference time.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run on
        num_iterations: Number of iterations for averaging

    Returns:
        Timing metrics
    """
    model.eval()

    # Warmup
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure time
    if device == 'cuda':
        torch.cuda.synchronize()

    import time
    times = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'throughput_fps': float(input_size[0] / (np.mean(times) / 1000))
    }
