"""
Grad-CAM implementation for model interpretability.

Reference: https://arxiv.org/abs/1610.02391
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

    Generates class activation maps showing which regions of the image
    were important for the model's prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: The model to explain
            target_layer: The convolutional layer to compute gradients for
                         (typically the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate class activation map.

        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index (if None, uses predicted class)

        Returns:
            CAM as numpy array [H, W]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_cam_batch(
        self,
        input_batch: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate CAMs for a batch of images.

        Args:
            input_batch: Input tensor [B, C, H, W]
            target_classes: List of target class indices

        Returns:
            List of CAMs
        """
        cams = []
        for i in range(input_batch.size(0)):
            target_class = target_classes[i] if target_classes else None
            cam = self.generate_cam(input_batch[i:i+1], target_class)
            cams.append(cam)
        return cams


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image.

    Args:
        image: Original image [H, W, 3] in range [0, 255]
        heatmap: Heatmap [H, W] in range [0, 1]
        alpha: Transparency of heatmap
        colormap: OpenCV colormap

    Returns:
        Overlaid image [H, W, 3]
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlaid = (alpha * heatmap_color + (1 - alpha) * image).astype(np.uint8)

    return overlaid


def visualize_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    target_layer: torch.nn.Module,
    save_path: Optional[str] = None,
    top_k: int = 3
) -> None:
    """
    Visualize Grad-CAM for top-k predictions.

    Args:
        model: The model
        image: Preprocessed input tensor [1, 3, H, W]
        original_image: Original image as numpy array [H, W, 3]
        class_names: List of class names
        target_layer: Target layer for Grad-CAM
        save_path: Path to save visualization
        top_k: Number of top predictions to visualize
    """
    # Get predictions
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    # Get top-k predictions
    top_probs, top_indices = probs[0].topk(top_k)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Create figure
    fig, axes = plt.subplots(2, top_k + 1, figsize=(4 * (top_k + 1), 8))

    # Show original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Generate and display CAMs for top-k predictions
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        # Generate CAM
        cam = gradcam.generate_cam(image, target_class=int(idx))

        # Overlay on original image
        overlaid = overlay_heatmap(original_image, cam, alpha=0.5)

        # Display
        col = i + 1

        # Heatmap
        axes[0, col].imshow(cam, cmap='jet')
        axes[0, col].set_title(
            f'{class_names[idx]}\n({prob:.2%})',
            fontweight='bold'
        )
        axes[0, col].axis('off')

        # Overlaid
        axes[1, col].imshow(overlaid)
        axes[1, col].set_title('Overlay', fontweight='bold')
        axes[1, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to {save_path}")

    plt.show()


def get_target_layer(model: torch.nn.Module, backbone: str) -> torch.nn.Module:
    """
    Get the appropriate target layer for Grad-CAM based on backbone.

    Args:
        model: The model
        backbone: Backbone name

    Returns:
        Target layer for Grad-CAM
    """
    if 'resnet' in backbone.lower():
        # For ResNet, use the last layer of the last block
        return model.model.layer4[-1]
    elif 'efficientnet' in backbone.lower():
        # For EfficientNet, use the last conv layer
        return model.model.conv_head
    elif 'vit' in backbone.lower():
        # For Vision Transformer, use the last block
        return model.model.blocks[-1].norm1
    else:
        # Default: try to find the last convolutional layer
        conv_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)
        if conv_layers:
            return conv_layers[-1]
        else:
            raise ValueError(f"Could not find appropriate layer for backbone: {backbone}")


def generate_gradcam_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    backbone: str,
    device: str,
    num_samples: int = 10,
    save_dir: Optional[str] = None
) -> None:
    """
    Generate Grad-CAM visualizations for multiple samples.

    Args:
        model: The model
        dataloader: Data loader
        class_names: List of class names
        backbone: Backbone name
        device: Device
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    model.eval()
    target_layer = get_target_layer(model, backbone)

    count = 0
    for images, labels, domains in dataloader:
        if count >= num_samples:
            break

        for i in range(images.size(0)):
            if count >= num_samples:
                break

            image = images[i:i+1].to(device)
            label = labels[i].item()
            domain = domains[i]

            # Convert to numpy for visualization
            original_image = images[i].permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = (original_image * std + mean)
            original_image = (original_image * 255).clip(0, 255).astype(np.uint8)

            # Save path
            if save_dir:
                save_path = f"{save_dir}/gradcam_{count}_{class_names[label]}_{domain}.png"
            else:
                save_path = None

            # Visualize
            print(f"\n[{count+1}/{num_samples}] Domain: {domain} | True Label: {class_names[label]}")
            visualize_gradcam(
                model, image, original_image,
                class_names, target_layer,
                save_path=save_path,
                top_k=3
            )

            count += 1

    print(f"\nGenerated Grad-CAM visualizations for {count} samples")
