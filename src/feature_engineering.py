"""
Feature engineering pipeline for the No-Show Prediction Engine.
Encapsulates all transformations used in training, for reuse in dashboard and inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def prepare_features(df, label_encoders=None, fit=True):
    """
    Prepare features from raw booking data for ML model consumption.

    Args:
        df: Raw booking DataFrame
        label_encoders: Dict of pre-fitted LabelEncoders (for inference). If None, new ones are created.
        fit: If True, fit label encoders. If False, only transform (for inference).

    Returns:
        X: Feature DataFrame ready for model
        y: Target Series (only if 'appointment_outcome' column exists)
        label_encoders: Dict of fitted LabelEncoders
    """
    data = df.copy()

    # ─── Derived Features ─────────────────────────────────────────────────
    # No-show ratio: historical no-show tendency
    data["no_show_ratio"] = data["past_no_show_count"] / (data["past_visit_count"] + 1)

    # Cancellation ratio: historical cancellation tendency
    data["cancellation_ratio"] = data["past_cancellation_count"] / (data["past_visit_count"] + 1)

    # Is new customer (first visit)
    data["is_new_customer"] = (data["past_visit_count"] <= 1).astype(int)

    # Is loyal customer (10+ visits)
    data["is_loyal_customer"] = (data["past_visit_count"] >= 10).astype(int)

    # Lead time buckets
    data["lead_time_bucket"] = pd.cut(
        data["booking_lead_time_days"],
        bins=[0, 1, 3, 7, 14, 30],
        labels=["same_day", "1-3_days", "4-7_days", "1-2_weeks", "2+_weeks"],
        include_lowest=True
    )

    # Hour buckets
    data["hour_bucket"] = pd.cut(
        data["appointment_hour"],
        bins=[8, 11, 14, 17, 21],
        labels=["morning", "midday", "afternoon", "evening"],
        include_lowest=True
    )

    # Age group
    data["age_group"] = pd.cut(
        data["customer_age"],
        bins=[17, 25, 35, 45, 65],
        labels=["18-25", "26-35", "36-45", "46+"],
        include_lowest=True
    )

    # ─── Encode Categoricals ──────────────────────────────────────────────
    categorical_cols = [
        "service_type", "payment_method", "branch", "day_of_week",
        "gender", "lead_time_bucket", "hour_bucket", "age_group"
    ]

    if label_encoders is None:
        label_encoders = {}

    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)
            if fit:
                le = LabelEncoder()
                data[col + "_encoded"] = le.fit_transform(data[col])
                label_encoders[col] = le
            else:
                le = label_encoders.get(col)
                if le is not None:
                    # Handle unseen labels gracefully
                    data[col + "_encoded"] = data[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    data[col + "_encoded"] = 0

    # ─── Select Feature Columns ───────────────────────────────────────────
    feature_cols = [
        "booking_lead_time_days",
        "appointment_hour",
        "past_visit_count",
        "past_cancellation_count",
        "past_no_show_count",
        "is_weekend",
        "customer_age",
        "no_show_ratio",
        "cancellation_ratio",
        "is_new_customer",
        "is_loyal_customer",
        "service_type_encoded",
        "payment_method_encoded",
        "branch_encoded",
        "day_of_week_encoded",
        "gender_encoded",
        "lead_time_bucket_encoded",
        "hour_bucket_encoded",
        "age_group_encoded",
    ]

    X = data[feature_cols]

    # Extract target if available
    y = None
    if "appointment_outcome" in data.columns:
        y = (data["appointment_outcome"] == "No-Show").astype(int)

    return X, y, label_encoders


def get_feature_names():
    """Return human-readable feature names for display."""
    return {
        "booking_lead_time_days": "Booking Lead Time (days)",
        "appointment_hour": "Appointment Hour",
        "past_visit_count": "Past Visit Count",
        "past_cancellation_count": "Past Cancellation Count",
        "past_no_show_count": "Past No-Show Count",
        "is_weekend": "Is Weekend",
        "customer_age": "Customer Age",
        "no_show_ratio": "Historical No-Show Ratio",
        "cancellation_ratio": "Historical Cancellation Ratio",
        "is_new_customer": "Is New Customer",
        "is_loyal_customer": "Is Loyal Customer",
        "service_type_encoded": "Service Type",
        "payment_method_encoded": "Payment Method",
        "branch_encoded": "Branch",
        "day_of_week_encoded": "Day of Week",
        "gender_encoded": "Gender",
        "lead_time_bucket_encoded": "Lead Time Bucket",
        "hour_bucket_encoded": "Hour Bucket",
        "age_group_encoded": "Age Group",
    }
