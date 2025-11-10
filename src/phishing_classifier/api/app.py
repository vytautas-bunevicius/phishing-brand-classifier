"""
FastAPI application for phishing brand classification.
"""

import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from phishing_classifier.config import get_config
from phishing_classifier.models import create_model
from phishing_classifier.preprocessing import get_transforms


# Global variables for model and config
model = None
config = None
transform = None
device = None


def load_model(checkpoint_path: str, backbone: str = "efficientnet_b3"):
    """Load trained model."""
    global model, config, transform, device

    config = get_config()
    config.model.backbone = backbone

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=False,
        dropout=config.model.dropout,
        checkpoint_path=checkpoint_path,
        device=device
    )
    model.eval()

    # Load transform
    transform = get_transforms(config.data.image_size, is_training=False)

    print(f"Model loaded successfully on {device}")


# Create FastAPI app
app = FastAPI(
    title="Phishing Brand Classifier API",
    description="Deep learning-based phishing website detection through screenshot analysis",
    version="0.1.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    # This should be configured via environment variables in production
    checkpoint_path = "models/checkpoints/best_model_efficientnet_b3.pth"
    if Path(checkpoint_path).exists():
        load_model(checkpoint_path)
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("Please load model using /load_model endpoint")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Phishing Brand Classifier API",
        "version": "0.1.0",
        "status": "running" if model is not None else "model not loaded"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.post("/load_model")
async def load_model_endpoint(
    checkpoint_path: str,
    backbone: str = "efficientnet_b3"
):
    """
    Load a model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        backbone: Backbone architecture name
    """
    try:
        load_model(checkpoint_path, backbone)
        return {
            "status": "success",
            "message": f"Model loaded from {checkpoint_path}",
            "backbone": backbone,
            "device": str(device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict brand from website screenshot.

    Args:
        file: Image file (PNG, JPEG)

    Returns:
        Prediction results with probabilities for all brands
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load model first using /load_model endpoint."
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess
        image_array = np.array(image)
        transformed = transform(image=image_array)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        # Inference
        start_time = time.time()

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        inference_time = (time.time() - start_time) * 1000  # ms

        # Get predictions
        probs = probabilities.cpu().numpy()
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # Prepare response
        predictions = {
            brand: float(prob)
            for brand, prob in zip(config.data.brands, probs)
        }

        # Sort predictions by probability
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        response = {
            "predicted_brand": config.data.brands[predicted_class],
            "confidence": confidence,
            "is_phishing": predicted_class != len(config.data.brands) - 1,  # Not "others"
            "all_predictions": predictions,
            "top_3_predictions": dict(sorted_predictions[:3]),
            "inference_time_ms": round(inference_time, 2)
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Predict brands for multiple screenshots.

    Args:
        files: List of image files

    Returns:
        List of predictions for each image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load model first using /load_model endpoint."
        )

    if len(files) > 32:
        raise HTTPException(
            status_code=400,
            detail="Maximum 32 images allowed per batch"
        )

    results = []

    for file in files:
        try:
            # Read and preprocess image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_array = np.array(image)
            transformed = transform(image=image_array)
            image_tensor = transformed['image'].unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

            # Get predictions
            probs = probabilities.cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            predictions = {
                brand: float(prob)
                for brand, prob in zip(config.data.brands, probs)
            }

            results.append({
                "filename": file.filename,
                "predicted_brand": config.data.brands[predicted_class],
                "confidence": confidence,
                "is_phishing": predicted_class != len(config.data.brands) - 1,
                "all_predictions": predictions
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse(content={"results": results})


@app.get("/brands")
async def get_brands():
    """Get list of supported brands."""
    if config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "brands": config.data.brands,
        "num_brands": len(config.data.brands)
    }


@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "backbone": config.model.backbone,
        "num_classes": config.model.num_classes,
        "input_size": config.data.image_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
