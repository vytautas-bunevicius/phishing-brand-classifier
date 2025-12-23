"""
Dataset class for phishing website screenshots.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PhishingDataset(Dataset):
    """
    Dataset for phishing website screenshots.

    Expected directory structure:
        data_dir/
            brand1/
                domain1.png
                domain2.png
            brand2/
                domain3.png
            ...
    """

    def __init__(
        self,
        data_dir: Path,
        brands: List[str],
        transform: Optional[Callable] = None,
        image_paths: Optional[List[Path]] = None,
        labels: Optional[List[int]] = None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing brand folders
            brands: List of brand names
            transform: Optional transform to apply to images
            image_paths: Optional pre-defined image paths
            labels: Optional pre-defined labels
        """
        self.data_dir = Path(data_dir)
        self.brands = brands
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(brands)}
        self.transform = transform

        if image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
        else:
            self.image_paths, self.labels = self._load_dataset()

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """Load all image paths and their labels."""
        image_paths = []
        labels = []

        for brand in self.brands:
            brand_dir = self.data_dir / brand
            if not brand_dir.exists():
                print(f"Warning: Brand directory not found: {brand_dir}")
                continue

            # Get all image files
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                for img_path in brand_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(self.brand_to_idx[brand])

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get item by index.

        Returns:
            image: Transformed image tensor
            label: Brand label
            domain: Domain name from filename
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Extract domain from filename
        domain = img_path.stem

        return image, label, domain

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across brands."""
        distribution = {}
        for brand in self.brands:
            count = sum(1 for label in self.labels if label == self.brand_to_idx[brand])
            distribution[brand] = count
        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.
        Uses inverse frequency weighting.
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)


def create_dataloaders(
    data_dir: Path,
    brands: List[str],
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Root directory containing brand folders
        brands: List of brand names
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        seed: Random seed

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        class_distribution: Distribution of classes
    """
    # Load full dataset to get all paths and labels
    full_dataset = PhishingDataset(data_dir, brands)

    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # Get class distribution
    class_distribution = full_dataset.get_class_distribution()
    print(f"\nClass distribution:")
    for brand, count in class_distribution.items():
        print(f"  {brand}: {count}")

    # Split data maintaining class balance
    image_paths = full_dataset.image_paths
    labels = full_dataset.labels

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=seed
    )

    # Second split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=seed
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_paths)} samples")
    print(f"  Val: {len(val_paths)} samples")
    print(f"  Test: {len(test_paths)} samples")

    # Create datasets
    train_dataset = PhishingDataset(
        data_dir, brands, train_transform, train_paths, train_labels
    )
    val_dataset = PhishingDataset(
        data_dir, brands, val_transform, val_paths, val_labels
    )
    test_dataset = PhishingDataset(
        data_dir, brands, val_transform, test_paths, test_labels
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_distribution
