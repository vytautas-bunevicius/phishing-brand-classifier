"""Model interpretability and explainability utilities.

This module provides tools to understand model decision-making:
- GradCAM for visual explanations
- Feature attribution using Captum
- Confidence analysis
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import AlbumentationsTransform, get_val_transforms


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Provides visual explanations for CNN decisions by highlighting
    regions of the input image that are important for predictions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize GradCAM.

        Args:
            model: The classifier model.
            target_layer: The convolutional layer to use for CAM.
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate GradCAM heatmap.

        Args:
            input_tensor: Input tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Compute weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Resize to input size
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def generate_heatmap_overlay(
        self,
        image: Union[np.ndarray, Image.Image],
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Generate heatmap overlay on the original image.

        Args:
            image: Original image as numpy array or PIL Image.
            cam: GradCAM heatmap.
            alpha: Overlay transparency.
            colormap: Matplotlib colormap name.

        Returns:
            Overlay image as numpy array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Resize CAM to image size
        from PIL import Image as PILImage

        cam_resized = np.array(
            PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), PILImage.Resampling.BILINEAR
            )
        ) / 255.0

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)

        # Create overlay
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)

        return overlay


class IntegratedGradients:
    """Integrated Gradients for feature attribution.

    Computes attributions by integrating gradients along a path
    from a baseline to the input.
    """

    def __init__(self, model: nn.Module):
        """Initialize Integrated Gradients.

        Args:
            model: The classifier model.
        """
        self.model = model

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> np.ndarray:
        """Generate integrated gradients attribution.

        Args:
            input_tensor: Input tensor of shape (1, C, H, W).
            target_class: Target class index.
            baseline: Baseline tensor (default: zeros).
            steps: Number of integration steps.

        Returns:
            Attribution array of shape (C, H, W).
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # Generate scaled inputs
        scaled_inputs = [
            baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad = True

        # Forward pass
        outputs = self.model(scaled_inputs)
        target_outputs = outputs[:, target_class]

        # Backward pass
        self.model.zero_grad()
        target_outputs.sum().backward()

        # Compute integrated gradients
        gradients = scaled_inputs.grad
        avg_gradients = gradients.mean(dim=0, keepdim=True)
        integrated_gradients = (input_tensor - baseline) * avg_gradients

        return integrated_gradients.squeeze().cpu().numpy()


class ModelExplainer:
    """High-level interface for model explanations."""

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        image_size: int = 224,
        device: str = "auto",
    ):
        """Initialize the explainer.

        Args:
            model: The classifier model.
            class_names: List of class names.
            image_size: Input image size.
            device: Device to use.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names
        self.image_size = image_size

        # Setup transform
        self.transform = AlbumentationsTransform(get_val_transforms(image_size=image_size))

        # Initialize GradCAM
        target_layer = model.target_layer
        if target_layer is not None:
            self.gradcam = GradCAM(model, target_layer)
        else:
            self.gradcam = None

        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(model)

    def preprocess(self, image: Union[str, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess an image.

        Args:
            image: Image path or PIL Image.

        Returns:
            Tuple of (tensor, original_image_array).
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        original = np.array(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        return tensor, original

    def explain(
        self,
        image: Union[str, Image.Image],
        target_class: Optional[int] = None,
        methods: List[str] = ["gradcam"],
    ) -> Dict:
        """Generate explanations for a prediction.

        Args:
            image: Image path or PIL Image.
            target_class: Target class to explain (default: predicted class).
            methods: List of explanation methods to use.

        Returns:
            Dictionary containing explanations.
        """
        tensor, original = self.preprocess(image)

        # Get prediction
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probs_np.max())

        if target_class is None:
            target_class = predicted_idx

        result = {
            "predicted_class": predicted_class,
            "target_class": self.class_names[target_class],
            "confidence": confidence,
            "all_probabilities": {
                name: float(probs_np[i]) for i, name in enumerate(self.class_names)
            },
            "original_image": original,
        }

        # Generate explanations
        if "gradcam" in methods and self.gradcam is not None:
            cam = self.gradcam.generate(tensor, target_class)
            overlay = self.gradcam.generate_heatmap_overlay(original, cam)
            result["gradcam"] = {
                "heatmap": cam,
                "overlay": overlay,
            }

        if "integrated_gradients" in methods:
            ig_attr = self.ig.generate(tensor, target_class)
            result["integrated_gradients"] = ig_attr

        return result

    def plot_explanation(
        self,
        explanation: Dict,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot explanation visualizations.

        Args:
            explanation: Explanation dictionary from explain().
            figsize: Figure size.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        num_plots = 1  # Original
        if "gradcam" in explanation:
            num_plots += 2  # Heatmap + overlay
        if "integrated_gradients" in explanation:
            num_plots += 1

        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Original image
        axes[plot_idx].imshow(explanation["original_image"])
        axes[plot_idx].set_title(
            f"Predicted: {explanation['predicted_class']}\n"
            f"Confidence: {explanation['confidence']:.3f}"
        )
        axes[plot_idx].axis("off")
        plot_idx += 1

        # GradCAM
        if "gradcam" in explanation:
            axes[plot_idx].imshow(explanation["gradcam"]["heatmap"], cmap="jet")
            axes[plot_idx].set_title("GradCAM Heatmap")
            axes[plot_idx].axis("off")
            plot_idx += 1

            axes[plot_idx].imshow(explanation["gradcam"]["overlay"])
            axes[plot_idx].set_title("GradCAM Overlay")
            axes[plot_idx].axis("off")
            plot_idx += 1

        # Integrated Gradients
        if "integrated_gradients" in explanation:
            ig = explanation["integrated_gradients"]
            # Take absolute value and sum across channels
            ig_vis = np.abs(ig).sum(axis=0)
            ig_vis = ig_vis / ig_vis.max()

            axes[plot_idx].imshow(ig_vis, cmap="hot")
            axes[plot_idx].set_title("Integrated Gradients")
            axes[plot_idx].axis("off")
            plot_idx += 1

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def analyze_misclassification(
        self,
        image: Union[str, Image.Image],
        true_class: str,
    ) -> Dict:
        """Analyze a misclassification case.

        Args:
            image: Image path or PIL Image.
            true_class: True class name.

        Returns:
            Analysis dictionary.
        """
        true_idx = self.class_names.index(true_class)

        # Get explanations for both predicted and true class
        tensor, original = self.preprocess(image)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())
        predicted_class = self.class_names[predicted_idx]

        analysis = {
            "true_class": true_class,
            "predicted_class": predicted_class,
            "is_correct": predicted_class == true_class,
            "true_class_probability": float(probs_np[true_idx]),
            "predicted_class_probability": float(probs_np[predicted_idx]),
            "probability_gap": float(probs_np[predicted_idx] - probs_np[true_idx]),
            "original_image": original,
        }

        if self.gradcam is not None:
            # GradCAM for predicted class
            cam_pred = self.gradcam.generate(tensor, predicted_idx)
            analysis["gradcam_predicted"] = self.gradcam.generate_heatmap_overlay(
                original, cam_pred
            )

            # GradCAM for true class
            cam_true = self.gradcam.generate(tensor, true_idx)
            analysis["gradcam_true"] = self.gradcam.generate_heatmap_overlay(
                original, cam_true
            )

        return analysis

    def get_feature_importance_summary(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100,
    ) -> Dict:
        """Get summary statistics of feature importance across samples.

        Args:
            dataloader: DataLoader with samples.
            num_samples: Number of samples to analyze.

        Returns:
            Summary statistics dictionary.
        """
        if self.gradcam is None:
            return {"error": "GradCAM not available for this model"}

        all_cams = []
        samples_processed = 0

        for images, labels, _ in dataloader:
            if samples_processed >= num_samples:
                break

            for i in range(min(images.size(0), num_samples - samples_processed)):
                tensor = images[i : i + 1].to(self.device)
                cam = self.gradcam.generate(tensor)
                all_cams.append(cam)
                samples_processed += 1

        # Compute statistics
        all_cams = np.array(all_cams)

        return {
            "mean_activation": all_cams.mean(axis=0),
            "std_activation": all_cams.std(axis=0),
            "num_samples": samples_processed,
        }
