"""Dataset classes for phishing brand classification."""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """Configuration for dataset creation using Pydantic v2."""

    data_dir: Path = Field(..., description="Root directory containing brand folders")
    train_split: float = Field(default=0.7, ge=0, le=1)
    val_split: float = Field(default=0.15, ge=0, le=1)
    test_split: float = Field(default=0.15, ge=0, le=1)
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    seed: int = Field(default=42)
    use_weighted_sampler: bool = Field(default=True)

    @field_validator("train_split", "val_split", "test_split", mode="after")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("Split must be between 0 and 1")
        return v


class PhishingDataset(Dataset):
    """Dataset for phishing website brand classification.

    Each sample consists of a website screenshot and its corresponding brand label.
    The 'others' class represents benign websites that don't belong to any targeted brand.

    Expected directory structure:
        data_dir/
            brand1/
                domain1.png
                domain2.png
            brand2/
                domain3.png
            others/
                benign1.png
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def __init__(
        self,
        data_dir: Path,
        brands: list[str],
        transform: Optional[Callable] = None,
        image_paths: Optional[list[Path]] = None,
        labels: Optional[list[int]] = None,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Root directory containing brand folders.
            brands: List of brand names in order.
            transform: Optional albumentations transform to apply to images.
            image_paths: Optional pre-defined image paths (for splits).
            labels: Optional pre-defined labels (for splits).
        """
        self.data_dir = Path(data_dir)
        self.brands = brands
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(brands)}
        self.idx_to_brand = {idx: brand for brand, idx in self.brand_to_idx.items()}
        self.transform = transform
        self.num_classes = len(brands)

        if image_paths is not None and labels is not None:
            self.image_paths = list(image_paths)
            self.labels = list(labels)
        else:
            self.image_paths, self.labels = self._scan_directory()

    def _scan_directory(self) -> tuple[list[Path], list[int]]:
        """Scan the data directory for images organized in brand folders."""
        image_paths: list[Path] = []
        labels: list[int] = []

        for brand in self.brands:
            brand_dir = self.data_dir / brand
            if not brand_dir.exists():
                logger.warning(f"Brand directory not found: {brand_dir}")
                continue

            for img_path in brand_dir.iterdir():
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    image_paths.append(img_path)
                    labels.append(self.brand_to_idx[brand])

        logger.info(f"Found {len(image_paths)} images across {len(self.brands)} brands")
        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_idx, domain_name)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image_array = np.array(image)

        if self.transform:
            transformed = self.transform(image=image_array)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        domain = img_path.stem

        return image_tensor, label, domain

    def get_class_distribution(self) -> dict[str, int]:
        """Get distribution of samples across brands."""
        distribution: dict[str, int] = {}
        for brand in self.brands:
            count = sum(1 for label in self.labels if label == self.brand_to_idx[brand])
            distribution[brand] = count
        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance.

        Uses inverse frequency weighting: weight = total / (num_classes * count)
        """
        class_counts = np.bincount(self.labels, minlength=self.num_classes)
        total_samples = len(self.labels)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        class_weights = total_samples / (self.num_classes * class_counts)
        return torch.FloatTensor(class_weights)

    def get_sample_weights(self) -> list[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [float(class_weights[label]) for label in self.labels]


def create_dataloaders(
    data_dir: Path,
    brands: list[str],
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Create train, validation, and test dataloaders.

    Args:
        data_dir: Root directory containing brand folders.
        brands: List of brand names.
        train_transform: Albumentations transform for training data.
        val_transform: Albumentations transform for validation/test data.
        batch_size: Batch size for dataloaders.
        num_workers: Number of data loading workers.
        train_split: Proportion of data for training.
        val_split: Proportion of data for validation.
        test_split: Proportion of data for testing.
        use_weighted_sampler: Whether to use weighted sampling for class imbalance.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_distribution)

    Raises:
        ValueError: If no images are found in the data directory.
    """
    full_dataset = PhishingDataset(data_dir, brands)

    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    class_distribution = full_dataset.get_class_distribution()
    logger.info(f"Class distribution: {class_distribution}")

    image_paths = full_dataset.image_paths
    labels = full_dataset.labels

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    logger.info(
        f"Dataset splits - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}"
    )

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

    return train_loader, val_loader, test_loader, class_distribution


def scan_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Scan the dataset directory and create a DataFrame of all images.

    Args:
        data_dir: Root directory containing brand folders.

    Returns:
        DataFrame with image information including path, label, dimensions, etc.
    """
    data_path = Path(data_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    records: list[dict] = []
    for brand_dir in data_path.iterdir():
        if brand_dir.is_dir():
            brand_name = brand_dir.name
            for img_path in brand_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            mode = img.mode
                    except Exception:
                        width, height, mode = None, None, None

                    records.append(
                        {
                            "image_path": str(img_path),
                            "label": brand_name,
                            "filename": img_path.name,
                            "domain": img_path.stem,
                            "file_size": img_path.stat().st_size,
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "extension": img_path.suffix.lower(),
                        }
                    )

    df = pd.DataFrame(records)
    logger.info(f"Found {len(df)} images across {df['label'].nunique()} brands")
    return df
