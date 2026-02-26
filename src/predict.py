"""
Prediction module for No-Show risk scoring.
Loads the trained model and provides prediction interface.
"""

import joblib
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_model(model_path=None):
    """Load the best trained model."""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    return joblib.load(model_path)


def load_label_encoders(encoders_path=None):
    """Load fitted label encoders."""
    if encoders_path is None:
        encoders_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    return joblib.load(encoders_path)


def get_risk_label(probability):
    """Convert probability to business-friendly risk label."""
    if probability >= 0.6:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"


def get_risk_color(label):
    """Get color for risk label (for dashboard use)."""
    return {"High": "#FF4444", "Medium": "#FFA500", "Low": "#44BB44"}.get(label, "#888888")


def predict_no_show(model, X):
    """
    Predict no-show probability for booking(s).

    Args:
        model: Trained model
        X: Feature DataFrame (from feature_engineering.prepare_features)

    Returns:
        DataFrame with probability and risk_label columns
    """
    probabilities = model.predict_proba(X)[:, 1]
    risk_labels = [get_risk_label(p) for p in probabilities]

    return pd.DataFrame({
        "no_show_probability": probabilities,
        "risk_label": risk_labels,
    })
