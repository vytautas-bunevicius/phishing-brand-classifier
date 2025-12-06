"""Image transforms for training and evaluation."""

from typing import Dict, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


def get_train_transforms(
    image_size: int = 224,
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """Get training transforms with data augmentation.

    Args:
        image_size: Target image size.
        augmentation_config: Optional config dict for augmentation parameters.

    Returns:
        Albumentations compose transform.
    """
    config = augmentation_config or {}

    return A.Compose(
        [
            # Resize to target size
            A.Resize(image_size, image_size),
            # Horizontal flip
            A.HorizontalFlip(p=0.5 if config.get("horizontal_flip", True) else 0),
            # Rotation
            A.Rotate(
                limit=config.get("rotation_limit", 15),
                p=0.5,
                border_mode=0,
            ),
            # Color augmentations
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=config.get("brightness_limit", 0.2),
                        contrast_limit=config.get("contrast_limit", 0.2),
                        p=1.0,
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            # Blur and noise
            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=config.get("blur_limit", 3),
                        p=1.0,
                    ),
                    A.GaussNoise(
                        std_range=(0.1, 0.3),
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            # Quality degradation (simulates different screenshot qualities)
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(60, 100), p=1.0),
                    A.Downscale(scale_range=(0.5, 0.9), p=1.0),
                ],
                p=0.3,
            ),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # Convert to tensor
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Albumentations compose transform.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(image_size: int = 224, num_augments: int = 5) -> list:
    """Get test-time augmentation transforms.

    Args:
        image_size: Target image size.
        num_augments: Number of augmented versions to create.

    Returns:
        List of transform compositions.
    """
    tta_transforms = [
        # Original
        get_val_transforms(image_size),
        # Horizontal flip
        A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        ),
        # Slight rotation left
        A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Rotate(limit=(5, 5), p=1.0, border_mode=0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        ),
        # Slight rotation right
        A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Rotate(limit=(-5, -5), p=1.0, border_mode=0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        ),
        # Scale variation
        A.Compose(
            [
                A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                A.CenterCrop(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        ),
    ]

    return tta_transforms[:num_augments]


class AlbumentationsTransform:
    """Wrapper to use Albumentations transforms with PyTorch Dataset."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transform to PIL Image.

        Args:
            image: PIL Image.

        Returns:
            Transformed tensor.
        """
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        return transformed["image"]


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor image for visualization.

    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
        Denormalized tensor with values in [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    tensor = tensor * std.to(tensor.device) + mean.to(tensor.device)
    return torch.clamp(tensor, 0, 1)
