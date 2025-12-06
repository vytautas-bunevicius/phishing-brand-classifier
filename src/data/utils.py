"""Data utilities for downloading and preparing the dataset."""

import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def download_dataset(
    destination: str = "data/raw",
    s3_bucket: str = "phishing-detection-homework-public-bucket",
) -> bool:
    """Download the dataset from S3.

    Args:
        destination: Local directory to save the dataset.
        s3_bucket: S3 bucket name.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    command = [
        "aws",
        "s3",
        "cp",
        f"s3://{s3_bucket}",
        str(dest_path),
        "--recursive",
        "--no-sign-request",
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Dataset downloaded successfully to {destination}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr}")
        return False
    except FileNotFoundError:
        print("AWS CLI not found. Please install it with: pip install awscli")
        return False


def scan_dataset(data_dir: str) -> pd.DataFrame:
    """Scan the dataset directory and create a DataFrame of all images.

    Args:
        data_dir: Root directory containing brand folders.

    Returns:
        DataFrame with image information.
    """
    data_path = Path(data_dir)
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    records = []
    for brand_dir in data_path.iterdir():
        if brand_dir.is_dir():
            brand_name = brand_dir.name
            for img_path in brand_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    # Try to get image dimensions
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
    print(f"Found {len(df)} images across {df['label'].nunique()} brands")
    return df


def prepare_dataset_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, validation, and test sets.

    Args:
        df: DataFrame with image data.
        train_size: Proportion for training.
        val_size: Proportion for validation.
        test_size: Proportion for testing.
        stratify: Whether to stratify by label.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    stratify_col = df["label"] if stratify else None

    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    # Second split: train vs val
    val_ratio = val_size / (train_size + val_size)
    stratify_col = train_val_df["label"] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=stratify_col,
        random_state=random_state,
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get the distribution of samples across classes.

    Args:
        df: DataFrame with 'label' column.

    Returns:
        Dictionary mapping class names to counts.
    """
    return df["label"].value_counts().to_dict()


def analyze_image_properties(df: pd.DataFrame) -> Dict:
    """Analyze image properties in the dataset.

    Args:
        df: DataFrame with image metadata.

    Returns:
        Dictionary with analysis results.
    """
    analysis = {
        "total_images": len(df),
        "num_classes": df["label"].nunique(),
        "class_distribution": get_class_distribution(df),
        "file_size_stats": {
            "min": df["file_size"].min(),
            "max": df["file_size"].max(),
            "mean": df["file_size"].mean(),
            "median": df["file_size"].median(),
        },
        "extension_distribution": df["extension"].value_counts().to_dict(),
    }

    if "width" in df.columns and df["width"].notna().any():
        analysis["width_stats"] = {
            "min": df["width"].min(),
            "max": df["width"].max(),
            "mean": df["width"].mean(),
        }
        analysis["height_stats"] = {
            "min": df["height"].min(),
            "max": df["height"].max(),
            "mean": df["height"].mean(),
        }

    return analysis


def create_sample_dataset(
    output_dir: str = "data/raw",
    num_samples_per_class: int = 100,
    image_size: Tuple[int, int] = (1920, 1080),
    random_seed: int = 42,
) -> None:
    """Create a sample dataset for testing purposes.

    This generates synthetic images to test the pipeline when
    the actual dataset is not available.

    Args:
        output_dir: Directory to save sample images.
        num_samples_per_class: Number of samples per class.
        image_size: Size of generated images.
        random_seed: Random seed.
    """
    np.random.seed(random_seed)
    output_path = Path(output_dir)

    # Brand colors for synthetic data
    brand_colors = {
        "amazon": (255, 153, 0),
        "apple": (100, 100, 100),
        "facebook": (59, 89, 152),
        "google": (66, 133, 244),
        "linkedin": (0, 119, 181),
        "microsoft": (0, 120, 212),
        "netflix": (229, 9, 20),
        "paypal": (0, 48, 135),
        "instagram": (225, 48, 108),
        "twitter": (29, 161, 242),
        "others": (128, 128, 128),
    }

    for brand, base_color in brand_colors.items():
        brand_dir = output_path / brand
        brand_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            # Create synthetic image with brand color + noise
            img_array = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)
            img_array[:, :] = base_color

            # Add some noise and variation
            noise = np.random.randint(-30, 30, (image_size[1], image_size[0], 3))
            img_array = np.clip(img_array.astype(np.int32) + noise, 0, 255).astype(np.uint8)

            # Save image
            img = Image.fromarray(img_array)
            img_path = brand_dir / f"sample_{i:04d}.png"
            img.save(img_path)

    print(f"Created sample dataset in {output_dir}")


def validate_image(image_path: str) -> Tuple[bool, Optional[str]]:
    """Validate that an image file is readable and valid.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to ensure it's fully readable
        with Image.open(image_path) as img:
            img.load()
        return True, None
    except Exception as e:
        return False, str(e)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate all images in the dataset.

    Args:
        df: DataFrame with 'image_path' column.

    Returns:
        DataFrame with added 'is_valid' and 'error' columns.
    """
    results = df["image_path"].apply(lambda x: validate_image(x))
    df["is_valid"] = results.apply(lambda x: x[0])
    df["error"] = results.apply(lambda x: x[1])

    invalid_count = (~df["is_valid"]).sum()
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid images")

    return df
