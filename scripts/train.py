"""
Training script for phishing brand classifier.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from phishing_classifier.config import Config, get_config
from phishing_classifier.preprocessing import create_dataloaders, get_transforms
from phishing_classifier.models import create_model, get_loss_function
from phishing_classifier.evaluation import MetricsCalculator


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train(config: Config) -> None:
    """Main training function."""
    print("="*80)
    print("PHISHING BRAND CLASSIFIER - TRAINING")
    print("="*80)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Set seed
    set_seed(config.data.seed)

    # Create directories
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    # Get transforms
    train_transform = get_transforms(config.data.image_size, is_training=True)
    val_transform = get_transforms(config.data.image_size, is_training=False)

    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_distribution = create_dataloaders(
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

    # Get class weights
    class_weights = None
    if config.model.use_class_weights:
        class_weights = train_loader.dataset.get_class_weights().to(device)
        print(f"\nUsing class weights: {class_weights}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        device=device
    )

    # Loss function
    criterion = get_loss_function(
        loss_type=config.model.loss_type,
        class_weights=class_weights,
        alpha=config.model.focal_alpha,
        gamma=config.model.focal_gamma,
        label_smoothing=config.model.label_smoothing
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay
    )

    # Learning rate scheduler
    if config.model.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.model.epochs,
            eta_min=1e-6
        )
    elif config.model.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    # Tensorboard
    writer = SummaryWriter(config.logs_dir / f'run_{config.model.backbone}')

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    # Metrics calculator
    metrics_calc = MetricsCalculator(config.data.brands)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    for epoch in range(1, config.model.epochs + 1):
        # Train
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_results = validate(
            model, val_loader, criterion, device, epoch
        )

        # Calculate metrics
        val_metrics = metrics_calc.calculate_metrics(
            np.array(val_results['labels']),
            np.array(val_results['predictions']),
            np.array(val_results['probabilities'])
        )

        # Update history
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['accuracy'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])

        # Tensorboard logging
        writer.add_scalar('Loss/train', train_results['loss'], epoch)
        writer.add_scalar('Loss/val', val_results['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_results['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_results['accuracy'], epoch)
        writer.add_scalar('F1/val', val_metrics['macro_f1'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.model.epochs}")
        print(f"  Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['accuracy']:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['macro_f1']:.4f} | Val AUC: {val_metrics.get('macro_roc_auc', 0):.4f}")
        print(f"  Others FPR: {val_metrics.get('others_fpr', 0):.4f}")

        # Learning rate scheduling
        if config.model.scheduler == 'plateau':
            scheduler.step(val_results['loss'])
        else:
            scheduler.step()

        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            best_val_acc = val_results['accuracy']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
                'config': config
            }

            save_path = config.models_dir / f'best_model_{config.model.backbone}.pth'
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model to {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.model.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_results['loss'],
        'val_acc': val_results['accuracy'],
        'config': config
    }
    final_save_path = config.models_dir / f'final_model_{config.model.backbone}.pth'
    torch.save(final_checkpoint, final_save_path)

    # Save training history
    history_path = config.results_dir / f'training_history_{config.model.backbone}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    writer.close()

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"History saved to: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train phishing brand classifier')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['ce', 'focal', 'label_smoothing', 'weighted'],
                        help='Loss function')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Data directory')

    args = parser.parse_args()

    # Get config and update with args
    config = get_config()
    config.model.backbone = args.backbone
    config.model.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.model.learning_rate = args.lr
    config.model.loss_type = args.loss
    config.data.data_dir = Path(args.data_dir)

    # Train
    train(config)


if __name__ == '__main__':
    main()
