"""Visualization utilities for analysis and reporting."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        normalize: Whether to normalize values.
        figsize: Figure size.
        cmap: Colormap name.
        title: Plot title.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training curves (loss and metrics over epochs).

    Args:
        history: Dictionary with training history.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss curves
    if "train_loss" in history and history["train_loss"]:
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    if "train_acc" in history and history["train_acc"]:
        axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    if "val_acc" in history and history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 curves
    if "train_f1" in history and history["train_f1"]:
        axes[2].plot(epochs, history["train_f1"], "b-", label="Train F1", linewidth=2)
    if "val_f1" in history and history["val_f1"]:
        axes[2].plot(epochs, history["val_f1"], "r-", label="Val F1", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score Curves")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot class distribution as a bar chart.

    Args:
        class_counts: Dictionary mapping class names to counts.
        figsize: Figure size.
        title: Plot title.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Color 'others' differently
    colors = ["steelblue" if c != "others" else "coral" for c in classes]

    bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=1)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticklabels(classes, rotation=45, ha="right")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Target Brands"),
        Patch(facecolor="coral", edgecolor="black", label="Others (Benign)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    class_names: List[str],
    num_cols: int = 4,
    figsize_per_image: Tuple[float, float] = (3, 3.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot sample predictions with images.

    Args:
        images: List of image arrays.
        true_labels: True class names.
        pred_labels: Predicted class names.
        confidences: Prediction confidences.
        class_names: All class names.
        num_cols: Number of columns.
        figsize_per_image: Size per image.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    n_samples = len(images)
    num_rows = (n_samples + num_cols - 1) // num_cols

    figsize = (figsize_per_image[0] * num_cols, figsize_per_image[1] * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    if num_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx < n_samples:
            img = images[idx]
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            conf = confidences[idx]

            ax.imshow(img)
            ax.axis("off")

            # Color based on correctness
            color = "green" if true_label == pred_label else "red"
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.2f}"
            ax.set_title(title, fontsize=9, color=color)
        else:
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_false_positive_analysis(
    fp_analysis: Dict,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot false positive analysis for 'others' class.

    Args:
        fp_analysis: False positive analysis dictionary.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar chart of misclassification by brand
    brands = list(fp_analysis["brand_misclassification_counts"].keys())
    counts = list(fp_analysis["brand_misclassification_counts"].values())

    axes[0].barh(brands, counts, color="coral", edgecolor="black")
    axes[0].set_xlabel("False Positive Count")
    axes[0].set_ylabel("Brand")
    axes[0].set_title("'Others' Misclassified as Brand")
    axes[0].invert_yaxis()

    # Summary statistics
    stats_text = (
        f"Total 'Others' Samples: {fp_analysis['total_others_samples']}\n"
        f"False Positives: {fp_analysis['false_positive_count']}\n"
        f"False Positive Rate: {fp_analysis['false_positive_rate']:.2%}\n\n"
        f"FP Confidence Stats:\n"
        f"  Mean: {fp_analysis['fp_confidence_stats']['mean']:.3f}\n"
        f"  Std: {fp_analysis['fp_confidence_stats']['std']:.3f}\n"
        f"  Min: {fp_analysis['fp_confidence_stats']['min']:.3f}\n"
        f"  Max: {fp_analysis['fp_confidence_stats']['max']:.3f}"
    )

    axes[1].text(
        0.1, 0.5, stats_text,
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1].axis("off")
    axes[1].set_title("False Positive Statistics")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confidence_distribution(
    confidences: np.ndarray,
    correct_mask: np.ndarray,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confidence distribution for correct vs incorrect predictions.

    Args:
        confidences: Prediction confidences.
        correct_mask: Boolean mask for correct predictions.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    ax.hist(
        correct_conf, bins=50, alpha=0.7, label=f"Correct (n={len(correct_conf)})",
        color="green", density=True
    )
    ax.hist(
        incorrect_conf, bins=50, alpha=0.7, label=f"Incorrect (n={len(incorrect_conf)})",
        color="red", density=True
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-class precision, recall, and F1 scores.

    Args:
        metrics: Dictionary containing per-class metrics.
        class_names: List of class names.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    width = 0.25

    precision = metrics.get("precision_per_class", [0] * len(class_names))
    recall = metrics.get("recall_per_class", [0] * len(class_names))
    f1 = metrics.get("f1_per_class", [0] * len(class_names))

    ax.bar(x - width, precision, width, label="Precision", color="steelblue")
    ax.bar(x, recall, width, label="Recall", color="coral")
    ax.bar(x + width, f1, width, label="F1 Score", color="seagreen")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
