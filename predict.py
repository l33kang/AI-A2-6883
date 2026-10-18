#!/usr/bin/env python
"""
CLI predictor for Musanze Harvest Intelligence Service.
Loads saved artifacts and accepts one JSON record.
Returns yield estimate, risk prediction, probability, cluster, and verification code.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.validation import VERIFICATION_CODE, LLLL


def load_artifacts():
    """Load all saved models and preprocessors."""
    models_dir = Path(__file__).parent / "models"
    
    artifacts = {}
    
    # Load regression pipeline
    reg_path = models_dir / "regression_pipeline.joblib"
    if reg_path.exists():
        artifacts["regression"] = joblib.load(reg_path)
    else:
        raise FileNotFoundError(f"Regression pipeline not found at {reg_path}")
    
    # Load classification pipeline
    clf_path = models_dir / "classification_pipeline.joblib"
    if clf_path.exists():
        artifacts["classification"] = joblib.load(clf_path)
    else:
        raise FileNotFoundError(f"Classification pipeline not found at {clf_path}")
    
    # Load clustering pipeline
    clust_path = models_dir / "clustering_pipeline.joblib"
    if clust_path.exists():
        artifacts["clustering"] = joblib.load(clust_path)
    else:
        raise FileNotFoundError(f"Clustering pipeline not found at {clust_path}")
    
    return artifacts


def predict_from_json(json_path: Path, artifacts: dict) -> dict:
    """Make prediction from a JSON file."""
    # Load JSON
    with open(json_path) as f:
        data = json.load(f)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data])
    
    # Regression prediction
    yield_estimate = float(artifacts["regression"].predict(input_df)[0])
    
    # Classification prediction
    high_risk_pred = int(artifacts["classification"].predict(input_df)[0])
    high_risk_prob = 0.5
    clf_model = artifacts["classification"].named_steps["model"]
    if hasattr(clf_model, "predict_proba"):
        prob = artifacts["classification"].predict_proba(input_df)[0]
        if len(prob) > 1:
            high_risk_prob = float(prob[1])
    
    # Clustering prediction
    cluster = int(artifacts["clustering"].predict(input_df)[0])
    
    return {
        "yield_estimate_tons": round(yield_estimate, 4),
        "high_risk_prediction": high_risk_pred,
        "high_risk_probability": round(high_risk_prob, 4),
        "cluster": cluster,
        "verification_code": VERIFICATION_CODE,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict yield, risk, and cluster for a farm record")
    parser.add_argument("--json", required=True, help="Path to JSON file with farm data")
    args = parser.parse_args()
    
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        artifacts = load_artifacts()
        result = predict_from_json(json_path, artifacts)
        
        # Print as JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()