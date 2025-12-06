"""Training script for the phishing brand classifier."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.data.transforms import AlbumentationsTransform, get_train_transforms, get_val_transforms
from src.data.utils import prepare_dataset_splits, scan_dataset
from src.models.classifier import create_model
from src.models.losses import FocalLoss
from src.utils.metrics import MetricTracker, calculate_metrics, find_optimal_threshold


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Set up logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path / f"{experiment_name}.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_accumulation_steps: int = 1,
) -> Tuple[float, float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.
        scaler: Optional gradient scaler for mixed precision.
        gradient_accumulation_steps: Number of steps to accumulate gradients.

    Returns:
        Tuple of (average_loss, accuracy, f1_score).
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with optional mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * gradient_accumulation_steps

        # Collect predictions
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    avg_loss = running_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the model.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average_loss, accuracy, f1_score, all_labels, all_preds, all_probs).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = np.mean(all_preds == all_labels)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1, all_labels, all_preds, all_probs


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    class_names: list = None,
    config: dict = None,
):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_names": class_names,
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path / filename)


def train(
    config: Dict,
    data_dir: str,
    output_dir: str,
    experiment_name: Optional[str] = None,
) -> Dict:
    """Main training function.

    Args:
        config: Configuration dictionary.
        data_dir: Path to data directory.
        output_dir: Path to output directory.
        experiment_name: Optional experiment name.

    Returns:
        Dictionary with training results.
    """
    # Setup
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_dir)
    checkpoint_dir = output_path / "models" / experiment_name
    log_dir = output_path / "logs" / experiment_name

    logger = setup_logging(str(log_dir), experiment_name)
    logger.info(f"Starting training experiment: {experiment_name}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and prepare data
    logger.info("Loading dataset...")
    df = scan_dataset(data_dir)

    train_df, val_df, test_df = prepare_dataset_splits(
        df,
        train_size=config["data"]["train_split"],
        val_size=config["data"]["val_split"],
        test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"],
    )

    # Transforms
    train_transform = AlbumentationsTransform(
        get_train_transforms(
            image_size=config["data"]["image_size"],
            augmentation_config=config.get("augmentation", {}).get("train", {}),
        )
    )
    val_transform = AlbumentationsTransform(
        get_val_transforms(image_size=config["data"]["image_size"])
    )

    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        use_weighted_sampler=config["model"].get("use_class_weights", True),
        class_names=config["data"].get("classes"),
    )

    logger.info(f"Classes: {class_names}")
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    model = create_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)
    logger.info(f"Model: {config['model']['architecture']}")

    # Loss function
    class_weights = None
    if config["model"].get("use_class_weights", True):
        train_dataset = train_loader.dataset
        class_weights = train_dataset.get_class_weights().to(device)
        logger.info(f"Class weights: {class_weights.tolist()}")

    if config["model"].get("use_focal_loss", True):
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config["model"].get("focal_loss_gamma", 2.0),
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler_type = config["training"].get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=config["training"]["learning_rate"] / 100,
        )
    elif scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config["training"]["scheduler_step_size"],
            gamma=config["training"]["scheduler_gamma"],
        )
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    else:
        scheduler = None

    # Mixed precision training
    scaler = GradScaler() if config["training"].get("use_amp", True) and device.type == "cuda" else None

    # Tensorboard
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    # Training loop
    metric_tracker = MetricTracker()
    best_val_f1 = 0
    patience_counter = 0
    early_stopping_patience = config["training"].get("early_stopping_patience", 10)

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['num_epochs']}")

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        )

        # Validate
        val_loss, val_acc, val_f1, val_labels, val_preds, val_probs = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logger.info(
            f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
        )
        logger.info(
            f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Tensorboard logging
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalars("F1", {"train": train_f1, "val": val_f1}, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Track metrics
        metric_tracker.update(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            train_f1=train_f1,
            val_f1=val_f1,
            learning_rate=current_lr,
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_f1": val_f1, "val_acc": val_acc, "val_loss": val_loss},
                checkpoint_dir=str(checkpoint_dir),
                filename="best_model.pt",
                class_names=class_names,
                config=config,
            )
            logger.info(f"Saved best model with val_f1: {val_f1:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"val_f1": val_f1, "val_acc": val_acc, "val_loss": val_loss},
        checkpoint_dir=str(checkpoint_dir),
        filename="final_model.pt",
        class_names=class_names,
        config=config,
    )

    # Final evaluation on test set
    logger.info("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt")["model_state_dict"])

    test_loss, test_acc, test_f1, test_labels, test_preds, test_probs = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    # Calculate comprehensive metrics
    test_metrics = calculate_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_proba=test_probs,
        class_names=class_names,
    )

    logger.info("\nTest Results:")
    logger.info(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    logger.info(f"\n{test_metrics['classification_report']}")

    # Find optimal threshold
    others_idx = class_names.index("others") if "others" in class_names else -1
    if others_idx >= 0:
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            y_true=test_labels,
            y_proba=test_probs,
            others_class_idx=others_idx,
        )
        logger.info(f"\nOptimal threshold: {optimal_threshold}")
        logger.info(f"Metrics at threshold: {threshold_metrics}")

    writer.close()

    results = {
        "experiment_name": experiment_name,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "training_history": metric_tracker.get_history(),
        "checkpoint_dir": str(checkpoint_dir),
    }

    # Save results
    with open(checkpoint_dir / "results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train phishing brand classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Train
    results = train(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    print(f"\nTraining complete! Best val F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoints saved to: {results['checkpoint_dir']}")


if __name__ == "__main__":
    main()
