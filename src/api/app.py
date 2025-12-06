"""FastAPI application for serving the phishing brand classifier.

This API provides endpoints for:
- Single image classification
- Batch classification
- Model information
- Health checks
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from src.predict import PhishingClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier: Optional[PhishingClassifier] = None


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    predicted_class: str = Field(..., description="Final predicted class")
    raw_predicted_class: str = Field(
        ..., description="Raw predicted class before threshold"
    )
    confidence: float = Field(..., description="Prediction confidence", ge=0, le=1)
    is_confident: bool = Field(..., description="Whether confidence exceeds threshold")
    is_rejected: bool = Field(
        ..., description="Whether prediction was rejected due to low confidence"
    )
    confidence_threshold: float = Field(..., description="Applied confidence threshold")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse]
    total_processing_time_ms: float


class TopKPrediction(BaseModel):
    """Model for top-k prediction entry."""

    class_name: str = Field(..., description="Class name")
    probability: float = Field(..., description="Probability", ge=0, le=1)
    rank: int = Field(..., description="Rank (1-indexed)")


class TopKResponse(BaseModel):
    """Response model for top-k predictions."""

    predictions: List[TopKPrediction]
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    model_architecture: str
    num_classes: int
    class_names: List[str]
    confidence_threshold: float
    device: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    device: str


def load_model():
    """Load the model from checkpoint."""
    global classifier

    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT", "outputs/models/best_model.pt"
    )
    device = os.environ.get("DEVICE", "auto")
    threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return

    try:
        classifier = PhishingClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            confidence_threshold=threshold,
        )
        logger.info(f"Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    load_model()
    yield
    # Shutdown
    pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Phishing Brand Classifier API",
        description="API for classifying website screenshots into brand categories for phishing detection",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=classifier is not None,
            device=str(classifier.device) if classifier else "unknown",
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info():
        """Get model information."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return ModelInfoResponse(
            model_architecture=classifier.config.get("model", {}).get(
                "architecture", "unknown"
            ),
            num_classes=len(classifier.class_names),
            class_names=classifier.class_names,
            confidence_threshold=classifier.confidence_threshold,
            device=str(classifier.device),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        """Classify a single website screenshot.

        Args:
            file: Uploaded image file (PNG, JPG, JPEG, WebP)

        Returns:
            Prediction response with class and confidence
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate file type
        allowed_types = {"image/png", "image/jpeg", "image/webp"}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed_types}",
            )

        try:
            # Read and process image
            start_time = time.perf_counter()

            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Get prediction
            result = classifier.predict(image)

            processing_time = (time.perf_counter() - start_time) * 1000

            return PredictionResponse(
                predicted_class=result["predicted_class"],
                raw_predicted_class=result["raw_predicted_class"],
                confidence=result["confidence"],
                is_confident=result["is_confident"],
                is_rejected=result["is_rejected"],
                confidence_threshold=result["confidence_threshold"],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(files: List[UploadFile] = File(...)):
        """Classify multiple website screenshots.

        Args:
            files: List of uploaded image files

        Returns:
            Batch prediction response
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(files) > 100:
            raise HTTPException(
                status_code=400, detail="Maximum 100 images per batch"
            )

        start_time = time.perf_counter()
        predictions = []

        for file in files:
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")

                img_start = time.perf_counter()
                result = classifier.predict(image)
                img_time = (time.perf_counter() - img_start) * 1000

                predictions.append(
                    PredictionResponse(
                        predicted_class=result["predicted_class"],
                        raw_predicted_class=result["raw_predicted_class"],
                        confidence=result["confidence"],
                        is_confident=result["is_confident"],
                        is_rejected=result["is_rejected"],
                        confidence_threshold=result["confidence_threshold"],
                        processing_time_ms=img_time,
                    )
                )
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing {file.filename}: {e}"
                )

        total_time = (time.perf_counter() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions, total_processing_time_ms=total_time
        )

    @app.post("/predict/top-k", response_model=TopKResponse)
    async def predict_top_k(file: UploadFile = File(...), k: int = 3):
        """Get top-k predictions for a website screenshot.

        Args:
            file: Uploaded image file
            k: Number of top predictions to return (default: 3)

        Returns:
            Top-k predictions with probabilities
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if k < 1 or k > len(classifier.class_names):
            raise HTTPException(
                status_code=400,
                detail=f"k must be between 1 and {len(classifier.class_names)}",
            )

        try:
            start_time = time.perf_counter()

            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            top_k = classifier.get_top_k_predictions(image, k=k)

            processing_time = (time.perf_counter() - start_time) * 1000

            predictions = [
                TopKPrediction(
                    class_name=p["class"],
                    probability=p["probability"],
                    rank=p["rank"],
                )
                for p in top_k
            ]

            return TopKResponse(
                predictions=predictions, processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Top-k prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/with-explanation")
    async def predict_with_explanation(file: UploadFile = File(...)):
        """Get prediction with GradCAM explanation.

        Returns prediction along with attention heatmap coordinates
        indicating which regions influenced the decision.

        Args:
            file: Uploaded image file

        Returns:
            Prediction with explanation data
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Get prediction
            result = classifier.predict(image, return_all_probs=True)

            # Note: GradCAM visualization would be added here
            # For now, return prediction with probabilities

            return {
                "prediction": result,
                "explanation_available": False,
                "note": "GradCAM visualization endpoint - implementation requires additional setup",
            }

        except Exception as e:
            logger.error(f"Explanation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/benchmark")
    async def benchmark():
        """Run inference speed benchmark.

        Returns timing statistics for model inference.
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Create a dummy image for benchmarking
            dummy_image = Image.new("RGB", (1920, 1080), color="white")

            benchmark_results = classifier.benchmark_inference_speed(
                dummy_image, num_iterations=50, warmup_iterations=5
            )

            return benchmark_results

        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Create app instance
app = create_app()


def main():
    """Run the API server."""
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
