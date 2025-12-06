"""Benchmark comparison: Custom EfficientNet vs AutoGluon MultiModalPredictor."""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import PhishingDataset
from src.data.transforms import AlbumentationsTransform, get_val_transforms
from src.models.classifier import BrandClassifier


def benchmark_custom_model(test_df: pd.DataFrame, checkpoint_path: str, class_names: list) -> dict:
    """Benchmark our custom EfficientNet model."""
    print("\n" + "=" * 60)
    print("Benchmarking Custom EfficientNet Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = BrandClassifier(
        num_classes=len(class_names),
        architecture="efficientnet_b0",
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create dataset
    transform = AlbumentationsTransform(get_val_transforms(image_size=224))
    dataset = PhishingDataset(
        data_dir=str(project_root / "data" / "raw"),
        df=test_df,
        transform=transform,
        class_names=class_names,
    )

    # Run inference
    predictions = []
    true_labels = []
    inference_times = []

    with torch.no_grad():
        for i in range(len(dataset)):
            image, label, _ = dataset[i]
            image = image.unsqueeze(0).to(device)

            start_time = time.perf_counter()
            output = model(image)
            pred = output.argmax(dim=1).item()
            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            predictions.append(pred)
            true_labels.append(label)
            inference_times.append(inference_time)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    avg_latency = sum(inference_times) / len(inference_times)

    results = {
        "model": "Custom EfficientNet-B0",
        "accuracy": accuracy,
        "f1_weighted": f1,
        "avg_latency_ms": avg_latency,
        "samples": len(dataset),
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")

    return results


def benchmark_autogluon(train_df: pd.DataFrame, test_df: pd.DataFrame, class_names: list) -> dict:
    """Benchmark AutoGluon MultiModalPredictor."""
    print("\n" + "=" * 60)
    print("Benchmarking AutoGluon MultiModalPredictor")
    print("=" * 60)

    try:
        from autogluon.multimodal import MultiModalPredictor
    except ImportError:
        print("AutoGluon not installed. Skipping AutoGluon benchmark.")
        return None

    # Prepare data for AutoGluon
    train_data = train_df[["image_path", "label"]].copy()
    train_data.columns = ["image", "label"]

    test_data = test_df[["image_path", "label"]].copy()
    test_data.columns = ["image", "label"]

    # Create output directory
    ag_output = project_root / "outputs" / "autogluon_benchmark"
    ag_output.mkdir(parents=True, exist_ok=True)

    # Train AutoGluon model
    print("Training AutoGluon model (this may take a while)...")
    start_train = time.perf_counter()

    predictor = MultiModalPredictor(
        label="label",
        path=str(ag_output),
        problem_type="multiclass",
    )

    # Use corrected hyperparameter names
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            'model.timm_image.checkpoint_name': 'efficientnet_b0',
            'optim.max_epochs': 3,  # Corrected from 'optimization.max_epochs'
            'env.per_gpu_batch_size': 16,
        },
        time_limit=600,  # 10 minutes max
    )

    train_time = time.perf_counter() - start_train
    print(f"Training time: {train_time:.2f}s")

    # Evaluate
    print("Evaluating AutoGluon model...")

    # Run inference with timing
    inference_times = []
    predictions = []

    for _, row in test_data.iterrows():
        sample = pd.DataFrame([{"image": row["image"]}])

        start_time = time.perf_counter()
        pred = predictor.predict(sample)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        predictions.append(pred.iloc[0])
        inference_times.append(inference_time)

    # Calculate metrics
    true_labels = test_data["label"].tolist()
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    avg_latency = sum(inference_times) / len(inference_times)

    results = {
        "model": "AutoGluon MultiModalPredictor",
        "accuracy": accuracy,
        "f1_weighted": f1,
        "avg_latency_ms": avg_latency,
        "samples": len(test_data),
        "train_time_s": train_time,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")

    return results


def main():
    """Run benchmark comparison."""
    print("=" * 60)
    print("Model Benchmark Comparison")
    print("Custom EfficientNet vs AutoGluon")
    print("=" * 60)

    # Check for data
    processed_dir = project_root / "data" / "processed"
    train_csv = processed_dir / "train.csv"
    test_csv = processed_dir / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        print("Error: Processed data not found. Please run data preparation first.")
        return

    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Get class names
    class_names = sorted(train_df["label"].unique().tolist())
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Check for custom model checkpoint
    checkpoint_path = project_root / "outputs" / "models" / "best_model.pt"
    if not checkpoint_path.exists():
        # Try test_run subdirectory
        checkpoint_path = project_root / "outputs" / "models" / "test_run" / "best_model.pt"

    results = []

    # Benchmark custom model
    if checkpoint_path.exists():
        custom_results = benchmark_custom_model(test_df, str(checkpoint_path), class_names)
        results.append(custom_results)
    else:
        print(f"Warning: Custom model checkpoint not found at {checkpoint_path}")

    # Benchmark AutoGluon
    ag_results = benchmark_autogluon(train_df, test_df, class_names)
    if ag_results:
        results.append(ag_results)

    # Print comparison
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON SUMMARY")
        print("=" * 60)

        comparison_df = pd.DataFrame(results)
        print(comparison_df.to_string(index=False))

        # Save results
        output_path = project_root / "outputs" / "benchmark_results.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    elif len(results) == 1:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS (single model)")
        print("=" * 60)
        for key, value in results[0].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
