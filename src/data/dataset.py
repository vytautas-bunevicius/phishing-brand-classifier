"""Dataset classes for phishing brand classification."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class PhishingDataset(Dataset):
    """Dataset for phishing website brand classification.

    Each sample consists of a website screenshot and its corresponding brand label.
    The 'others' class represents benign websites that don't belong to any targeted brand.
    """

    def __init__(
        self,
        data_dir: str,
        df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Root directory containing brand folders.
            df: Optional DataFrame with 'image_path' and 'label' columns.
                If None, will scan data_dir for images.
            transform: Optional transform to apply to images.
            class_names: List of class names in order. If None, will be inferred.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        if df is not None:
            self.df = df.copy()
        else:
            self.df = self._scan_directory()

        # Set up class names and mapping
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = sorted(self.df["label"].unique().tolist())
            # Ensure 'others' is last if present
            if "others" in self.class_names:
                self.class_names.remove("others")
                self.class_names.append("others")

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_names)

        # Add numeric labels
        self.df["label_idx"] = self.df["label"].map(self.class_to_idx)

    def _scan_directory(self) -> pd.DataFrame:
        """Scan the data directory for images organized in brand folders."""
        records = []
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

        for brand_dir in self.data_dir.iterdir():
            if brand_dir.is_dir():
                brand_name = brand_dir.name
                for img_path in brand_dir.iterdir():
                    if img_path.suffix.lower() in valid_extensions:
                        records.append(
                            {
                                "image_path": str(img_path),
                                "label": brand_name,
                                "filename": img_path.name,
                                "domain": img_path.stem,
                            }
                        )

        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a single sample.

        Returns:
            Tuple of (image_tensor, label_idx, image_path)
        """
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label_idx = row["label_idx"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx, image_path

    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        return self.df["label"].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance.

        Uses inverse frequency weighting.
        """
        class_counts = self.df["label_idx"].value_counts().sort_index()
        total = len(self.df)
        weights = total / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = self.df["label_idx"].map(lambda x: class_weights[x].item())
        return torch.FloatTensor(sample_weights.values)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: str,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    class_names: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test dataloaders.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        data_dir: Root data directory.
        train_transform: Transform for training data.
        val_transform: Transform for validation/test data.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        use_weighted_sampler: Whether to use weighted sampling for class imbalance.
        class_names: Optional list of class names.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Create datasets
    train_dataset = PhishingDataset(
        data_dir=data_dir,
        df=train_df,
        transform=train_transform,
        class_names=class_names,
    )

    val_dataset = PhishingDataset(
        data_dir=data_dir,
        df=val_df,
        transform=val_transform,
        class_names=train_dataset.class_names,
    )

    test_dataset = PhishingDataset(
        data_dir=data_dir,
        df=test_df,
        transform=val_transform,
        class_names=train_dataset.class_names,
    )

    # Create sampler for class imbalance
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_names
