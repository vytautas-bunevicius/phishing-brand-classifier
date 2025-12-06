"""Inference script for the phishing brand classifier."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.data.transforms import AlbumentationsTransform, get_val_transforms
from src.models.classifier import BrandClassifier, create_model


class PhishingClassifier:
    """Inference wrapper for the phishing brand classifier."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        confidence_threshold: float = 0.85,
    ):
        """Initialize the classifier.

        Args:
            checkpoint_path: Path to model checkpoint.
            device: Device to use ('auto', 'cuda', 'cpu').
            confidence_threshold: Minimum confidence for predictions.
        """
        self.confidence_threshold = confidence_threshold

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.class_names = checkpoint.get("class_names", [])
        self.config = checkpoint.get("config", {})

        # Get model config
        model_config = self.config.get("model", {})
        data_config = self.config.get("data", {})

        # Create and load model
        self.model = BrandClassifier(
            architecture=model_config.get("architecture", "efficientnet_b0"),
            num_classes=model_config.get("num_classes", 11),
            pretrained=False,
            dropout=model_config.get("dropout", 0.3),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup transform
        image_size = data_config.get("image_size", 224)
        self.transform = AlbumentationsTransform(get_val_transforms(image_size=image_size))

        # Find 'others' class index
        self.others_idx = (
            self.class_names.index("others") if "others" in self.class_names else -1
        )

        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")

    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocess an image for inference.

        Args:
            image: Image path or PIL Image.

        Returns:
            Preprocessed tensor.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image type: {type(image)}")

        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        return_all_probs: bool = False,
    ) -> Dict:
        """Predict the brand for a single image.

        Args:
            image: Image path or PIL Image.
            return_all_probs: Whether to return probabilities for all classes.

        Returns:
            Prediction dictionary.
        """
        # Preprocess
        tensor = self.preprocess(image).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()[0]
        confidence = float(probs_np.max())
        predicted_idx = int(probs_np.argmax())
        predicted_class = self.class_names[predicted_idx]

        # Apply confidence threshold
        # If below threshold, classify as 'others' (benign) to minimize false positives
        is_confident = confidence >= self.confidence_threshold
        if not is_confident and self.others_idx >= 0:
            final_class = "others"
            is_rejected = True
        else:
            final_class = predicted_class
            is_rejected = False

        result = {
            "predicted_class": final_class,
            "raw_predicted_class": predicted_class,
            "confidence": confidence,
            "is_confident": is_confident,
            "is_rejected": is_rejected,
            "confidence_threshold": self.confidence_threshold,
        }

        if return_all_probs:
            result["all_probabilities"] = {
                name: float(probs_np[idx])
                for idx, name in enumerate(self.class_names)
            }

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> List[Dict]:
        """Predict brands for a batch of images.

        Args:
            images: List of image paths or PIL Images.
            batch_size: Batch size for inference.

        Returns:
            List of prediction dictionaries.
        """
        results = []

        # Process in batches
        for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
            batch_images = images[i : i + batch_size]

            # Preprocess batch
            tensors = []
            for img in batch_images:
                tensors.append(self.preprocess(img))
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=1)

            probs_np = probs.cpu().numpy()

            # Process each prediction
            for j in range(len(batch_images)):
                confidence = float(probs_np[j].max())
                predicted_idx = int(probs_np[j].argmax())
                predicted_class = self.class_names[predicted_idx]

                is_confident = confidence >= self.confidence_threshold
                if not is_confident and self.others_idx >= 0:
                    final_class = "others"
                    is_rejected = True
                else:
                    final_class = predicted_class
                    is_rejected = False

                results.append(
                    {
                        "predicted_class": final_class,
                        "raw_predicted_class": predicted_class,
                        "confidence": confidence,
                        "is_confident": is_confident,
                        "is_rejected": is_rejected,
                    }
                )

        return results

    def get_top_k_predictions(
        self,
        image: Union[str, Path, Image.Image],
        k: int = 3,
    ) -> List[Dict]:
        """Get top-k predictions for an image.

        Args:
            image: Image path or PIL Image.
            k: Number of top predictions to return.

        Returns:
            List of top-k prediction dictionaries.
        """
        tensor = self.preprocess(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()[0]
        top_indices = probs_np.argsort()[::-1][:k]

        return [
            {
                "class": self.class_names[idx],
                "probability": float(probs_np[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(top_indices)
        ]

    def benchmark_inference_speed(
        self,
        image: Union[str, Path, Image.Image],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict:
        """Benchmark inference speed.

        Args:
            image: Sample image for benchmarking.
            num_iterations: Number of iterations.
            warmup_iterations: Number of warmup iterations.

        Returns:
            Benchmark results dictionary.
        """
        tensor = self.preprocess(image).to(self.device)

        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(tensor)

        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # Convert to ms

        return {
            "device": str(self.device),
            "num_iterations": num_iterations,
            "mean_latency_ms": float(times.mean()),
            "std_latency_ms": float(times.std()),
            "min_latency_ms": float(times.min()),
            "max_latency_ms": float(times.max()),
            "p50_latency_ms": float(np.percentile(times, 50)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "p99_latency_ms": float(np.percentile(times, 99)),
            "throughput_fps": float(1000 / times.mean()),
        }


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Phishing brand classification inference")
    parser.add_argument(
        "image",
        type=str,
        nargs="+",
        help="Image path(s) to classify",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Return top-k predictions",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference speed benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Initialize classifier
    classifier = PhishingClassifier(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.threshold,
    )

    results = []

    for image_path in args.image:
        print(f"\nProcessing: {image_path}")

        if args.top_k:
            predictions = classifier.get_top_k_predictions(image_path, k=args.top_k)
            print(f"Top-{args.top_k} predictions:")
            for pred in predictions:
                print(f"  {pred['rank']}. {pred['class']}: {pred['probability']:.4f}")
            results.append({"image": image_path, "top_k": predictions})
        else:
            prediction = classifier.predict(image_path, return_all_probs=True)
            print(f"Predicted: {prediction['predicted_class']}")
            print(f"Confidence: {prediction['confidence']:.4f}")
            print(f"Is confident: {prediction['is_confident']}")
            results.append({"image": image_path, "prediction": prediction})

    # Benchmark if requested
    if args.benchmark:
        print("\nRunning inference speed benchmark...")
        benchmark = classifier.benchmark_inference_speed(args.image[0])
        print(f"Mean latency: {benchmark['mean_latency_ms']:.2f} ms")
        print(f"Throughput: {benchmark['throughput_fps']:.1f} FPS")
        results.append({"benchmark": benchmark})

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
