"""Quick demo: 3 example predictions using the trained model."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import joblib
import pandas as pd
from src.feature_engineering import prepare_features
from src.predict import predict_no_show

# Load model and encoders
model = joblib.load('models/best_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')


def run_prediction(label, details, booking_dict):
    df = pd.DataFrame([booking_dict])
    X, _, _ = prepare_features(df, label_encoders=encoders, fit=False)
    pred = predict_no_show(model, X)
    prob = pred["no_show_probability"].values[0]
    risk = pred["risk_label"].values[0]

    actions = {
        "High": "Require prepaid booking + double SMS reminder",
        "Medium": "Send reminder SMS 24h + 2h before appointment",
        "Low": "Standard confirmation SMS",
    }

    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    for line in details:
        print(f"  {line}")
    print("  ---")
    print(f"  No-Show Probability: {prob:.1%}")
    print(f"  Risk Label:          {risk}")
    print(f"  Action:              {actions.get(risk, 'N/A')}")
    print()
    return prob, risk


# === EXAMPLE 1: High-risk customer ===
p1, r1 = run_prediction(
    "EXAMPLE 1: Risky Customer",
    [
        "Customer:  CUST_0099 (Male, Age 22)",
        "Service:   Haircut at Whitefield",
        "Payment:   Cash at Salon",
        "Lead Time: 12 days ahead (Saturday)",
        "History:   5 visits, 4 no-shows, 3 cancellations",
    ],
    {
        "customer_id": "CUST_0099",
        "service_type": "Haircut",
        "branch": "Whitefield",
        "booking_lead_time_days": 12,
        "appointment_hour": 9,
        "past_visit_count": 5,
        "past_cancellation_count": 3,
        "past_no_show_count": 4,
        "payment_method": "Pay at Salon (Cash)",
        "day_of_week": "Saturday",
        "is_weekend": 1,
        "customer_age": 22,
        "gender": "Male",
        "booking_date": "2026-02-20",
        "appointment_date": "2026-03-04",
    },
)

# === EXAMPLE 2: Low-risk loyal customer ===
p2, r2 = run_prediction(
    "EXAMPLE 2: Loyal Customer",
    [
        "Customer:  CUST_0200 (Female, Age 30)",
        "Service:   Bridal Package at Koramangala",
        "Payment:   Online Prepaid",
        "Lead Time: 2 days ahead (Wednesday)",
        "History:   15 visits, 0 no-shows, 0 cancellations",
    ],
    {
        "customer_id": "CUST_0200",
        "service_type": "Bridal Package",
        "branch": "Koramangala",
        "booking_lead_time_days": 2,
        "appointment_hour": 14,
        "past_visit_count": 15,
        "past_cancellation_count": 0,
        "past_no_show_count": 0,
        "payment_method": "Online Prepaid",
        "day_of_week": "Wednesday",
        "is_weekend": 0,
        "customer_age": 30,
        "gender": "Female",
        "booking_date": "2026-02-24",
        "appointment_date": "2026-02-26",
    },
)

# === EXAMPLE 3: New customer ===
p3, r3 = run_prediction(
    "EXAMPLE 3: New Customer",
    [
        "Customer:  CUST_0500 (Female, Age 25)",
        "Service:   Spa & Massage at Indiranagar",
        "Payment:   Card on File",
        "Lead Time: 7 days ahead (Friday evening)",
        "History:   1 visit, 1 no-show, 1 cancellation",
    ],
    {
        "customer_id": "CUST_0500",
        "service_type": "Spa & Massage",
        "branch": "Indiranagar",
        "booking_lead_time_days": 7,
        "appointment_hour": 18,
        "past_visit_count": 1,
        "past_cancellation_count": 1,
        "past_no_show_count": 1,
        "payment_method": "Card on File",
        "day_of_week": "Friday",
        "is_weekend": 0,
        "customer_age": 25,
        "gender": "Female",
        "booking_date": "2026-02-19",
        "appointment_date": "2026-02-26",
    },
)

# Summary
print("=" * 60)
print("  SUMMARY — Model Predictions")
print("=" * 60)
print(f"  Risky customer:  {p1:.1%} -> {r1}")
print(f"  Loyal customer:  {p2:.1%} -> {r2}")
print(f"  New customer:    {p3:.1%} -> {r3}")
print("=" * 60)
