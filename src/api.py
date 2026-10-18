"""
FastAPI service module for Musanze Harvest Intelligence Service.
Provides GET /health, GET /model-info, POST /predict endpoints.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.validation import VERIFICATION_CODE, LLLL, EXPECTED_RANGES, EXPECTED_SEASONS


# Pydantic models for request/response validation
class FarmInput(BaseModel):
    """Input schema for a single farm prediction request."""
    rainfall_mm: float = Field(..., description="Rainfall in millimeters", ge=600, le=1400)
    avg_temp_c: float = Field(..., description="Average temperature in Celsius", ge=12, le=26)
    soil_moisture_pct: float = Field(..., description="Soil moisture percentage", ge=20, le=80)
    fertilizer_kg: float = Field(..., description="Fertilizer applied in kg", ge=0, le=250)
    field_size_ha: float = Field(..., description="Field size in hectares", ge=0.2, le=4.0)
    season: str = Field(..., description="Season: A or B")
    cooperative_support: int = Field(..., description="Cooperative support (0 or 1)", ge=0, le=1)
    
    @field_validator("season")
    @classmethod
    def validate_season(cls, v: str) -> str:
        if v not in EXPECTED_SEASONS:
            raise ValueError(f"season must be one of {EXPECTED_SEASONS}")
        return v
    
    @field_validator("cooperative_support")
    @classmethod
    def validate_cooperative(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError("cooperative_support must be 0 or 1")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    yield_estimate_tons: float = Field(..., description="Estimated yield in tons")
    high_risk_prediction: int = Field(..., description="High risk prediction (0 or 1)")
    high_risk_probability: float = Field(..., description="Probability of high risk")
    cluster: int = Field(..., description="Assigned cluster ID")
    verification_code: str = Field(..., description="Verification code")


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint."""
    verification_code: str
    seed: int
    regression_model: str
    classification_model: str
    clustering_k: int
    n_features: int
    feature_names: list


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""
    status: str
    verification_code: str


# Global model storage
_models = {}


def load_models() -> Dict[str, Any]:
    """Load all saved models and artifacts."""
    models_dir = Path(__file__).parent.parent / "models"
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    
    models = {}
    
    # Load regression pipeline
    reg_path = models_dir / "regression_pipeline.joblib"
    if reg_path.exists():
        models["regression"] = joblib.load(reg_path)
    
    # Load classification pipeline
    clf_path = models_dir / "classification_pipeline.joblib"
    if clf_path.exists():
        models["classification"] = joblib.load(clf_path)
    
    # Load clustering pipeline
    clust_path = models_dir / "clustering_pipeline.joblib"
    if clust_path.exists():
        models["clustering"] = joblib.load(clust_path)
    
    # Load metrics for model info
    reg_metrics_path = artifacts_dir / "regression_metrics.json"
    if reg_metrics_path.exists():
        with open(reg_metrics_path) as f:
            models["reg_metrics"] = json.load(f)
    
    clf_metrics_path = artifacts_dir / "classification_metrics.json"
    if clf_metrics_path.exists():
        with open(clf_metrics_path) as f:
            models["clf_metrics"] = json.load(f)
    
    clust_metrics_path = artifacts_dir / "clustering_report.json"
    if clust_metrics_path.exists():
        with open(clust_metrics_path) as f:
            models["clust_metrics"] = json.load(f)
    
    return models


def get_model_info() -> Dict[str, Any]:
    """Get model information for /model-info endpoint."""
    if not _models:
        load_models_into_global()
    
    reg_metrics = _models.get("reg_metrics", {})
    clf_metrics = _models.get("clf_metrics", {})
    clust_metrics = _models.get("clust_metrics", {})
    
    # Get feature names from regression pipeline
    feature_names = []
    if "regression" in _models:
        try:
            feature_names = _models["regression"].named_steps["preprocessor"].get_feature_names_out().tolist()
        except Exception:
            pass
    
    return {
        "verification_code": VERIFICATION_CODE,
        "seed": LLLL,
        "regression_model": reg_metrics.get("selected_model_type", "Unknown"),
        "classification_model": clf_metrics.get("selected_model_type", "Unknown"),
        "clustering_k": clust_metrics.get("selected_k", 0),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }


def load_models_into_global():
    """Load models into global storage."""
    global _models
    _models = load_models()


def predict_single(farm_data: FarmInput) -> PredictionResponse:
    """Make prediction for a single farm."""
    if not _models:
        load_models_into_global()
    
    # Convert to DataFrame for pipeline
    input_dict = farm_data.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Regression prediction
    if "regression" not in _models:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    yield_estimate = float(_models["regression"].predict(input_df)[0])
    
    # Classification prediction
    if "classification" not in _models:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    high_risk_pred = int(_models["classification"].predict(input_df)[0])
    high_risk_prob = 0.5
    if hasattr(_models["classification"].named_steps["model"], "predict_proba"):
        prob = _models["classification"].predict_proba(input_df)[0]
        if len(prob) > 1:
            high_risk_prob = float(prob[1])
    
    # Clustering prediction
    if "clustering" not in _models:
        raise HTTPException(status_code=503, detail="Clustering model not loaded")
    cluster = int(_models["clustering"].predict(input_df)[0])
    
    return PredictionResponse(
        yield_estimate_tons=round(yield_estimate, 4),
        high_risk_prediction=high_risk_pred,
        high_risk_probability=round(high_risk_prob, 4),
        cluster=cluster,
        verification_code=VERIFICATION_CODE,
    )


# Create FastAPI app
app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description="AI service for potato yield estimation, risk classification, and farm clustering",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models_into_global()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", verification_code=VERIFICATION_CODE)


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict(farm: FarmInput):
    """Predict yield, risk, and cluster for a farm."""
    try:
        return predict_single(farm)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Import pandas here to avoid circular import
import pandas as pd

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)