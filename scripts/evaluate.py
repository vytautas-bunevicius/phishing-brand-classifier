"""
Evaluation script for phishing brand classifier.
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from phishing_classifier.config import Config, get_config
from phishing_classifier.preprocessing import create_dataloaders, get_transforms
from phishing_classifier.models import create_model
from phishing_classifier.evaluation import (
    MetricsCalculator,
    calculate_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_per_class_metrics,
    plot_error_analysis
)
from phishing_classifier.evaluation.metrics import calculate_inference_time


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    config: Config
) -> Dict:
    """
    Evaluate model on test set.

    Returns:
        Dictionary containing all evaluation results
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_domains = []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels, domains in tqdm(test_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_domains.extend(domains)

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_calc = MetricsCalculator(config.data.brands)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_proba)

    # Get classification report
    classification_report = metrics_calc.generate_classification_report(y_true, y_pred)

    # Calculate optimal threshold for false positive reduction
    others_idx = len(config.data.brands) - 1
    optimal_threshold, threshold_metrics = calculate_optimal_threshold(
        y_true, y_proba, others_idx, target_fpr=config.evaluation.target_fpr
    )

    # Measure inference speed
    print("\nMeasuring inference speed...")
    timing_metrics = calculate_inference_time(
        model,
        input_size=(config.data.batch_size, 3, *config.data.image_size),
        device=device,
        num_iterations=100
    )

    # Compile results
    results = {
        'metrics': metrics,
        'classification_report': classification_report,
        'optimal_threshold': threshold_metrics,
        'inference_time': timing_metrics,
        'predictions': {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist(),
            'domains': all_domains
        }
    }

    return results


def print_results(results: Dict, config: Config) -> None:
    """Print evaluation results."""
    metrics = results['metrics']

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print("\n📊 Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  ROC-AUC (macro): {metrics.get('macro_roc_auc', 0):.4f}")

    print("\n🎯 False Positive Analysis ('others' class):")
    print(f"  Total 'others' samples: {metrics.get('others_total', 0)}")
    print(f"  Correctly classified: {metrics.get('others_correct', 0)}")
    print(f"  False positives: {metrics.get('others_false_positives', 0)}")
    print(f"  FPR (others): {metrics.get('others_fpr', 0):.4f}")
    print(f"  Target FPR: {config.evaluation.target_fpr}")

    if metrics.get('others_misclassified_as'):
        print("\n  'Others' misclassified as:")
        for brand, count in sorted(
            metrics['others_misclassified_as'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"    {brand}: {count}")

    print("\n⚡ Inference Speed:")
    timing = results['inference_time']
    print(f"  Mean: {timing['mean_ms']:.2f} ms")
    print(f"  Std: {timing['std_ms']:.2f} ms")
    print(f"  Min: {timing['min_ms']:.2f} ms")
    print(f"  Max: {timing['max_ms']:.2f} ms")
    print(f"  Throughput: {timing['throughput_fps']:.2f} images/sec")

    print("\n📋 Classification Report:")
    print(results['classification_report'])

    print("\n🎚️  Optimal Threshold (for target FPR):")
    opt_thresh = results['optimal_threshold']
    print(f"  Threshold: {opt_thresh['threshold']:.4f}")
    print(f"  Achieved FPR: {opt_thresh['fpr']:.4f}")
    print(f"  TPR: {opt_thresh['tpr']:.4f}")


def save_results(results: Dict, config: Config, backbone: str) -> None:
    """Save evaluation results."""
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON (excluding predictions to keep file size small)
    metrics_to_save = {
        'metrics': results['metrics'],
        'optimal_threshold': results['optimal_threshold'],
        'inference_time': results['inference_time']
    }

    # Convert numpy arrays in metrics to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    metrics_to_save = convert_to_serializable(metrics_to_save)

    metrics_path = results_dir / f'evaluation_metrics_{backbone}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")

    # Save classification report
    report_path = results_dir / f'classification_report_{backbone}.txt'
    with open(report_path, 'w') as f:
        f.write(results['classification_report'])
    print(f"✓ Saved classification report to {report_path}")

    # Generate and save visualizations
    print("\n📊 Generating visualizations...")

    y_true = np.array(results['predictions']['y_true'])
    y_pred = np.array(results['predictions']['y_pred'])
    y_proba = np.array(results['predictions']['y_proba'])

    # Confusion matrix
    cm_path = results_dir / f'confusion_matrix_{backbone}.png'
    plot_confusion_matrix(
        results['metrics']['confusion_matrix'],
        config.data.brands,
        save_path=cm_path,
        normalize=True
    )

    # ROC curves
    roc_path = results_dir / f'roc_curves_{backbone}.png'
    plot_roc_curves(
        y_true, y_proba,
        config.data.brands,
        save_path=roc_path
    )

    # Per-class metrics
    for metric_name in ['precision', 'recall', 'f1']:
        metric_key = f'per_class_{metric_name}'
        if metric_key in results['metrics']:
            metric_path = results_dir / f'per_class_{metric_name}_{backbone}.png'
            plot_per_class_metrics(
                results['metrics'][metric_key],
                metric_name.capitalize(),
                save_path=metric_path
            )

    # Error analysis
    error_path = results_dir / f'error_analysis_{backbone}.png'
    plot_error_analysis(
        y_true, y_pred,
        config.data.brands,
        save_path=error_path
    )

    print("\n✓ All visualizations saved!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate phishing brand classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                        help='Backbone architecture')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()

    print("="*80)
    print("PHISHING BRAND CLASSIFIER - EVALUATION")
    print("="*80)

    # Get config
    config = get_config()
    config.model.backbone = args.backbone
    config.data.data_dir = Path(args.data_dir)
    config.data.batch_size = args.batch_size

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=False,
        dropout=config.model.dropout,
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Create dataloaders
    print("\nLoading dataset...")
    train_transform = get_transforms(config.data.image_size, is_training=True)
    val_transform = get_transforms(config.data.image_size, is_training=False)

    _, _, test_loader, _ = create_dataloaders(
        data_dir=config.data.data_dir,
        brands=config.data.brands,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        seed=config.data.seed
    )

    # Evaluate
    results = evaluate_model(model, test_loader, device, config)

    # Print results
    print_results(results, config)

    # Save results
    save_results(results, config, args.backbone)

    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
