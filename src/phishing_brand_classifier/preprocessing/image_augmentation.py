"""
Image transformations and augmentation for phishing detection.
"""

from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True
) -> A.Compose:
    """
    Get image transformations.

    Args:
        image_size: Target image size (height, width)
        is_training: Whether this is for training (includes augmentation)

    Returns:
        Composed transformations
    """
    if is_training:
        # Training transforms with augmentation
        transform = A.Compose([
            # Resize
            A.Resize(height=image_size[0], width=image_size[1]),

            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            # Color augmentations (subtle to preserve brand colors)
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.5),

            # Blur and noise (simulate different screen captures)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),

            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # Compression artifacts (common in web screenshots)
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),

            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    return transform


def get_tta_transforms(
    image_size: Tuple[int, int] = (224, 224),
    num_augmentations: int = 5
) -> A.Compose:
    """
    Get test-time augmentation transforms.

    Args:
        image_size: Target image size
        num_augmentations: Number of augmented versions to create

    Returns:
        TTA transformations
    """
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),

        # Light augmentations for TTA
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
        ], p=0.5),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])

    return transform
