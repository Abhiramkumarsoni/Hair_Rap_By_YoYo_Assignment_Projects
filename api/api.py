"""
FastAPI Prediction API for No-Show Risk Scoring.
Hair Rap by YoYo — Salon Booking Platform.

Endpoints:
    GET  /         → Health check
    POST /predict  → Predict no-show risk & get recommended action
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import prepare_features
from src.predict import get_risk_label


# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hair Rap by YoYo — No-Show Prediction API",
    description="Predict salon appointment no-show risk and get recommended actions.",
    version="1.0.0",
)


# ─── Load Model ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

try:
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        model_metadata = json.load(f)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠ Warning: Could not load model artifacts: {e}")
    model = label_encoders = None
    model_metadata = {}


# ─── Schemas ─────────────────────────────────────────────────────────────────
class BookingRequest(BaseModel):
    """Booking details for prediction."""
    customer_id: str = Field(..., example="CUST_0042")
    service_type: str = Field(..., example="Hair Coloring")
    branch: str = Field(..., example="Koramangala")
    booking_lead_time_days: int = Field(..., ge=0, example=5)
    appointment_hour: int = Field(..., ge=9, le=20, example=14)
    past_visit_count: int = Field(..., ge=0, example=8)
    past_cancellation_count: int = Field(..., ge=0, example=1)
    past_no_show_count: int = Field(..., ge=0, example=2)
    payment_method: str = Field(..., example="Online Prepaid")
    day_of_week: str = Field(..., example="Wednesday")
    customer_age: int = Field(..., ge=18, le=80, example=28)
    gender: str = Field(..., example="Female")


class PredictionResponse(BaseModel):
    """Prediction result."""
    customer_id: str
    no_show_probability: float
    risk_label: str
    recommended_action: str


# ─── Helper Functions ────────────────────────────────────────────────────────
def get_recommended_action(risk_label: str) -> str:
    """Map risk level to a business action."""
    actions = {
        "High": "⚠️ Require prepaid booking. Send double SMS reminder (24h + 2h before). Consider overbooking the slot.",
        "Medium": "📱 Send reminder SMS 24h + 2h before appointment. Add to waitlist backup.",
        "Low": "✅ Standard confirmation. No special action needed.",
    }
    return actions.get(risk_label, "No action defined")


def booking_to_dataframe(booking: BookingRequest) -> pd.DataFrame:
    """Convert request to DataFrame for model input."""
    is_weekend = 1 if booking.day_of_week in ["Saturday", "Sunday"] else 0
    return pd.DataFrame([{
        "booking_id": "API_REQUEST",
        "customer_id": booking.customer_id,
        "service_type": booking.service_type,
        "booking_date": "2026-01-01",
        "appointment_date": "2026-01-01",
        "appointment_hour": booking.appointment_hour,
        "booking_lead_time_days": booking.booking_lead_time_days,
        "past_visit_count": booking.past_visit_count,
        "past_cancellation_count": booking.past_cancellation_count,
        "past_no_show_count": booking.past_no_show_count,
        "payment_method": booking.payment_method,
        "branch": booking.branch,
        "day_of_week": booking.day_of_week,
        "is_weekend": is_weekend,
        "customer_age": booking.customer_age,
        "gender": booking.gender,
    }])


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check."""
    return {
        "service": "Hair Rap by YoYo — No-Show Prediction API",
        "status": "healthy" if MODEL_LOADED else "degraded",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingRequest):
    """Predict no-show risk for a booking and return the recommended action."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        df = booking_to_dataframe(booking)
        X, _, _ = prepare_features(df, label_encoders=label_encoders, fit=False)

        probability = float(model.predict_proba(X)[0][1])
        risk_label = get_risk_label(probability)

        return PredictionResponse(
            customer_id=booking.customer_id,
            no_show_probability=round(probability, 4),
            risk_label=risk_label,
            recommended_action=get_recommended_action(risk_label),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting No-Show Prediction API...")
    print("📖 Docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
