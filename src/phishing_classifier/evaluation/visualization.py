"""
Visualization utilities for model evaluation and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix with improved visualization.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row (true labels)
    """
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        title = 'Normalized Confusion Matrix'
        fmt = '.2%'
        cm_display = cm_normalized
    else:
        title = 'Confusion Matrix (Counts)'
        fmt = 'd'
        cm_display = cm

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot ROC curves for all classes.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities [num_samples, num_classes]
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            label=f'{class_name} (AUC = {roc_auc:.3f})',
            linewidth=2
        )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")

    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: tuple = (15, 5)
) -> None:
    """
    Plot training history (loss and metrics).

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot F1 score
    if 'train_f1' in history and 'val_f1' in history:
        axes[2].plot(history['train_f1'], label='Train F1', linewidth=2)
        axes[2].plot(history['val_f1'], label='Val F1', linewidth=2)
        axes[2].set_xlabel('Epoch', fontweight='bold')
        axes[2].set_ylabel('F1 Score', fontweight='bold')
        axes[2].set_title('Training and Validation F1 Score', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")

    plt.show()


def plot_class_distribution(
    class_distribution: Dict[str, int],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot class distribution.

    Args:
        class_distribution: Dictionary mapping class names to counts
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())

    colors = sns.color_palette("husl", len(classes))
    bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontweight='bold'
        )

    ax.set_xlabel('Brand', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")

    plt.show()


def plot_per_class_metrics(
    metrics: Dict[str, float],
    metric_name: str,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot per-class metrics (precision, recall, F1).

    Args:
        metrics: Dictionary mapping class names to metric values
        metric_name: Name of the metric
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(metrics.keys())
    values = list(metrics.values())

    colors = sns.color_palette("viridis", len(classes))
    bars = ax.barh(classes, values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            value, bar.get_y() + bar.get_height() / 2.,
            f'{value:.3f}',
            ha='left', va='center',
            fontweight='bold'
        )

    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Brand', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class {metric_name} to {save_path}")

    plt.show()


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot error analysis showing misclassification patterns.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Error counts per class
    error_counts = []
    for i in range(len(class_names)):
        errors = cm[i, :].sum() - cm[i, i]
        error_counts.append(errors)

    colors = ['red' if ec > 0 else 'green' for ec in error_counts]
    ax1.barh(class_names, error_counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Errors', fontweight='bold')
    ax1.set_ylabel('True Class', fontweight='bold')
    ax1.set_title('Misclassification Counts by Class', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Error rates per class
    error_rates = []
    for i in range(len(class_names)):
        total = cm[i, :].sum()
        if total > 0:
            error_rate = (total - cm[i, i]) / total
            error_rates.append(error_rate)
        else:
            error_rates.append(0)

    colors = ['red' if er > 0.1 else 'orange' if er > 0.05 else 'green' for er in error_rates]
    ax2.barh(class_names, error_rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Error Rate', fontweight='bold')
    ax2.set_ylabel('True Class', fontweight='bold')
    ax2.set_title('Error Rates by Class', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")

    plt.show()
