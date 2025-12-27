"""
FastAPI application for phishing brand classification (2025 best practices).

Features:
- Pydantic v2 models for request/response validation
- Structured logging
- Proper error handling
- Health checks with detailed status
- API versioning
- OpenAPI documentation
"""

import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import structlog
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, field_validator
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from phishing_classifier.config import get_settings, Settings
from phishing_classifier.models import create_model
from phishing_classifier.preprocessing import get_transforms


# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if get_settings().log_format == "console" else structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


# ============================================================================
# Pydantic Models for API
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for prediction (if using JSON body)."""
    image_base64: str = Field(..., description="Base64 encoded image")


class BrandPrediction(BaseModel):
    """Single brand prediction result."""
    brand: str = Field(..., description="Predicted brand name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

    @field_validator('confidence')
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 4 decimal places."""
        return round(v, 4)


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    predicted_brand: str = Field(..., description="Most likely brand")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    is_phishing: bool = Field(..., description="Whether site is likely phishing")
    all_predictions: dict[str, float] = Field(..., description="All brand probabilities")
    top_3_predictions: list[BrandPrediction] = Field(..., description="Top 3 predictions")
    inference_time_ms: float = Field(..., gt=0, description="Inference time in milliseconds")
    model_version: str = Field(default="1.0.0", description="Model version")


class BatchPredictionResult(BaseModel):
    """Single result in batch prediction."""
    filename: str = Field(..., description="Original filename")
    predicted_brand: str = Field(..., description="Predicted brand")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    is_phishing: bool = Field(..., description="Whether site is likely phishing")
    all_predictions: dict[str, float] = Field(..., description="All brand probabilities")
    error: str | None = Field(default=None, description="Error message if failed")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    results: list[BatchPredictionResult] = Field(..., description="Prediction results")
    total_inference_time_ms: float = Field(..., gt=0, description="Total processing time")
    successful: int = Field(..., ge=0, description="Number of successful predictions")
    failed: int = Field(..., ge=0, description="Number of failed predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str | None = Field(default=None, description="Computation device")
    gpu_available: bool = Field(..., description="GPU availability")
    model_version: str | None = Field(default=None, description="Loaded model version")


class ModelInfo(BaseModel):
    """Model information response."""
    backbone: str = Field(..., description="Model architecture")
    num_classes: int = Field(..., description="Number of output classes")
    input_size: tuple[int, int] = Field(..., description="Expected input size")
    total_parameters: int = Field(..., description="Total model parameters")
    trainable_parameters: int = Field(..., description="Trainable parameters")
    device: str = Field(..., description="Current device")
    brands: list[str] = Field(..., description="Supported brands")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Global application state."""
    model: torch.nn.Module | None = None
    config: Settings | None = None
    transform: callable | None = None
    device: str | None = None
    model_version: str = "1.0.0"


state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)."""
    # Startup
    logger.info("Starting Phishing Classifier API")

    # Load settings
    state.config = get_settings()

    # Try to load model if checkpoint exists
    checkpoint_path = state.config.models_dir / f"best_model_{state.config.model.backbone}.pth"
    if checkpoint_path.exists():
        try:
            await load_model_internal(str(checkpoint_path), state.config.model.backbone)
            logger.info("Model loaded successfully on startup", checkpoint=str(checkpoint_path))
        except Exception as e:
            logger.warning("Failed to load model on startup", error=str(e))
    else:
        logger.warning("No model checkpoint found on startup", path=str(checkpoint_path))

    yield

    # Shutdown
    logger.info("Shutting down Phishing Classifier API")
    if state.model is not None:
        del state.model
        if state.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Phishing Brand Classifier API",
    description="Production-ready deep learning system for detecting phishing websites through screenshot analysis",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

async def load_model_internal(checkpoint_path: str, backbone: str) -> None:
    """Load model internally."""
    state.config = get_settings()
    state.config.model.backbone = backbone

    state.device = state.config.device_auto

    # Load model
    state.model = create_model(
        backbone=state.config.model.backbone,
        num_classes=state.config.data.num_classes,
        pretrained=False,
        dropout=state.config.model.dropout,
        checkpoint_path=checkpoint_path,
        device=state.device
    )
    state.model.eval()

    # Load transform
    state.transform = get_transforms(state.config.data.image_size, is_training=False)

    logger.info("Model loaded", backbone=backbone, device=state.device)


async def predict_image(image: Image.Image) -> dict:
    """Predict brand from image."""
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please load model first."
        )

    # Preprocess
    image_array = np.array(image)
    transformed = state.transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0).to(state.device)

    # Inference
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = state.model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Get predictions
    probs = probabilities.cpu().numpy()
    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    # Prepare response
    all_predictions = {
        brand: float(prob)
        for brand, prob in zip(state.config.data.brands, probs)
    }

    # Top 3
    top_indices = np.argsort(probs)[-3:][::-1]
    top_3 = [
        BrandPrediction(
            brand=state.config.data.brands[idx],
            confidence=float(probs[idx])
        )
        for idx in top_indices
    ]

    return {
        "predicted_brand": state.config.data.brands[predicted_idx],
        "confidence": confidence,
        "is_phishing": predicted_idx != len(state.config.data.brands) - 1,  # Not "others"
        "all_predictions": all_predictions,
        "top_3_predictions": top_3,
        "inference_time_ms": round(inference_time, 2),
        "model_version": state.model_version
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "Phishing Brand Classifier API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": state.model is not None,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health() -> HealthResponse:
    """Health check endpoint with detailed status."""
    return HealthResponse(
        status="healthy" if state.model is not None else "model_not_loaded",
        model_loaded=state.model is not None,
        device=state.device,
        gpu_available=torch.cuda.is_available(),
        model_version=state.model_version if state.model is not None else None
    )


@app.post("/load_model", tags=["Model Management"])
async def load_model(
    checkpoint_path: str = Field(..., description="Path to model checkpoint"),
    backbone: str = Field(default="efficientnet_b3", description="Model backbone")
) -> dict:
    """Load or reload model from checkpoint."""
    try:
        await load_model_internal(checkpoint_path, backbone)

        return {
            "status": "success",
            "message": f"Model loaded from {checkpoint_path}",
            "backbone": backbone,
            "device": state.device
        }
    except Exception as e:
        logger.error("Failed to load model", error=str(e), checkpoint=checkpoint_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict brand from website screenshot.

    Upload a screenshot image and get brand classification with confidence scores.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (PNG, JPEG, etc.)"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Predict
        result = await predict_image(image)

        logger.info(
            "Prediction completed",
            filename=file.filename,
            predicted=result["predicted_brand"],
            confidence=result["confidence"]
        )

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="List of screenshot images")
) -> BatchPredictionResponse:
    """
    Predict brands for multiple screenshots in batch.

    Maximum batch size is configurable (default: 32 images).
    """
    if len(files) > state.config.api.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {state.config.api.max_batch_size} images allowed per batch"
        )

    results = []
    start_time = time.perf_counter()
    successful = 0
    failed = 0

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')

            result = await predict_image(image)

            results.append(BatchPredictionResult(
                filename=file.filename,
                predicted_brand=result["predicted_brand"],
                confidence=result["confidence"],
                is_phishing=result["is_phishing"],
                all_predictions=result["all_predictions"]
            ))
            successful += 1

        except Exception as e:
            logger.warning("Batch prediction failed for file", filename=file.filename, error=str(e))
            results.append(BatchPredictionResult(
                filename=file.filename,
                predicted_brand="error",
                confidence=0.0,
                is_phishing=False,
                all_predictions={},
                error=str(e)
            ))
            failed += 1

    total_time = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Batch prediction completed",
        total_files=len(files),
        successful=successful,
        failed=failed,
        total_time_ms=total_time
    )

    return BatchPredictionResponse(
        results=results,
        total_inference_time_ms=round(total_time, 2),
        successful=successful,
        failed=failed
    )


@app.get("/brands", tags=["Information"])
async def get_brands() -> dict[str, list[str] | int]:
    """Get list of supported brands."""
    if state.config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    return {
        "brands": state.config.data.brands,
        "num_brands": state.config.data.num_classes
    }


@app.get("/model_info", response_model=ModelInfo, tags=["Information"])
async def get_model_info() -> ModelInfo:
    """Get detailed information about the loaded model."""
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    total_params = sum(p.numel() for p in state.model.parameters())
    trainable_params = sum(p.numel() for p in state.model.parameters() if p.requires_grad)

    return ModelInfo(
        backbone=state.config.model.backbone,
        num_classes=state.config.data.num_classes,
        input_size=state.config.data.image_size,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        device=state.device,
        brands=state.config.data.brands
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
        log_level=settings.log_level.lower(),
    )
