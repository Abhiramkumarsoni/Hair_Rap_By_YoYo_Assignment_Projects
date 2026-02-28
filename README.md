# 💇 AI-Powered No-Show & Customer Intelligence Module

### Hair Rap by YoYo — Enterprise Salon Booking Platform

An end-to-end AI module that predicts appointment no-shows, provides executive dashboards, identifies churn-risk customers, and demonstrates production-ready thinking.

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Dataset & EDA Findings](#-dataset--eda-findings)
- [Feature Engineering](#-feature-engineering)
- [Model Comparison & Results](#-model-comparison--results)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [FastAPI Prediction API](#-fastapi-prediction-api)
- [Business Logic & Decision Thresholds](#-business-logic--decision-thresholds)
- [Dashboard](#-ai-dashboard)
- [Retention Intelligence](#-retention-intelligence)
- [Production Architecture](#-production-architecture)
- [Assumptions](#-assumptions)

---

## 🎯 Problem Statement

Customers book salon appointments but often **don't show up**, causing:

- **Revenue loss** — time slots go to waste with no payment
- **Staff idle time** — stylists/barbers sit unproductive
- **Scheduling inefficiency** — other customers could have taken those slots

**Solution**: An AI-driven intelligence layer that:

1. **Predicts** appointment no-show probability before the appointment
2. **Converts** predictions into business actions (overbooking, reminders, prepaid requirements)
3. **Provides** interactive executive dashboards for decision-making
4. **Identifies** churn-risk customers with data-backed retention strategies

---

## 📁 Project Structure

```
Hair Rap by YoYo/
├── data/
│   ├── generate_mock_data.py       # Synthetic dataset generator
│   ├── salon_bookings.csv          # 10,000 booking records
│   └── customer_profiles.csv       # 2,000 customer profiles
├── src/
│   ├── feature_engineering.py      # Feature pipeline (19 features)
│   ├── train.py                    # Model training with MLflow tracking
│   ├── train_without_mlflow.py     # Standalone training (no MLflow)
│   ├── predict.py                  # Inference module
│   └── retention.py                # Customer segmentation & strategies
├── api/
│   └── api.py                      # FastAPI REST API for predictions
├── models/
│   ├── best_model.pkl              # Best trained model (Random Forest)
│   ├── label_encoders.pkl          # Fitted encoders for inference
│   ├── feature_importance.csv      # Feature importance rankings
│   ├── model_comparison.csv        # All model metrics
│   ├── model_metadata.json         # Training metadata
│   └── confusion_matrix.csv        # Best model confusion matrix
├── mlruns/                          # MLflow experiment tracking data
├── dashboard/
│   ├── app.py                      # Streamlit interactive dashboard
│   └── api_frontend.py             # API testing frontend (Streamlit)
├── experiment.ipynb                 # XGBoost experiment notebook
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (optional — already included)
python data/generate_mock_data.py

# 3. Train models (with MLflow tracking)
python src/train.py

# 4. Launch dashboard
streamlit run dashboard/app.py

# 5. Start prediction API (separate terminal)
uvicorn api.api:app --host 0.0.0.0 --port 8000

# 6. View MLflow experiment UI (separate terminal)
mlflow ui --backend-store-uri mlruns
```

---

## 📊 Dataset & EDA Findings

### Dataset Overview

- **10,000 bookings** across **6 months** (Aug 2025 – Feb 2026)
- **~1,990 unique customers** across **5 branches** (Bangalore)
- **6 service types**: Haircut, Hair Coloring, Spa & Massage, Facial, Bridal Package, Hair Treatment
- **Overall No-Show Rate**: ~23.5%

### Key EDA Insights

| Insight | Finding |
|---|---|
| **Payment Method** | Cash payments have **32.5% no-show rate** vs 17.7% for online prepaid |
| **Branch Variance** | Whitefield has the highest no-show rate (**28.8%**), Indiranagar lowest (**21.5%**) |
| **Service Impact** | Bridal Package has lowest no-show (**17.5%**) — high-value = committed |
| **Lead Time** | Bookings made >14 days ahead have significantly higher no-show rates |
| **Customer History** | Customers with past no-shows are 2-3x more likely to no-show again |
| **Time of Day** | Early morning (9-10 AM) and late evening (7-8 PM) slots have elevated no-show rates |
| **Weekend Effect** | Weekends show slightly higher no-show rates than weekdays |

---

## ⚙️ Feature Engineering

**19 engineered features** from raw booking data:

| Category | Features | Description |
|---|---|---|
| **Historical Behavior** | `no_show_ratio`, `cancellation_ratio` | Past no-show and cancellation tendencies |
| **Customer Profile** | `is_new_customer`, `is_loyal_customer`, `customer_age`, `gender` | Customer demographics and loyalty |
| **Temporal** | `appointment_hour`, `day_of_week`, `is_weekend`, `hour_bucket` | Time-based patterns |
| **Booking Context** | `booking_lead_time_days`, `lead_time_bucket` | How far in advance was the booking |
| **Service & Location** | `service_type`, `branch`, `payment_method` | Categorical features (label encoded) |

**Key derived features**:

- `no_show_ratio = past_no_show_count / (past_visit_count + 1)` — strongest predictor
- `cancellation_ratio = past_cancellation_count / (past_visit_count + 1)`
- `is_new_customer` (≤1 prior visit) and `is_loyal_customer` (≥10 prior visits)

---

## 🏆 Model Comparison & Results

| Model | Accuracy | Precision | Recall | F1 Score | **AUC-ROC** |
|---|---|---|---|---|---|
| **Random Forest** 🏆 | 0.6630 | 0.3117 | 0.3567 | 0.3327 | **0.6111** |
| LightGBM | 0.6155 | 0.3008 | 0.4777 | 0.3692 | 0.5946 |
| XGBoost | 0.6375 | 0.3016 | 0.4098 | 0.3474 | 0.5809 |
| Logistic Regression | 0.5585 | 0.2741 | 0.5308 | 0.3615 | 0.5646 |

> **Selected Model**: Random Forest (best AUC-ROC: 0.6111)
>
> All models use **class_weight='balanced'** or **scale_pos_weight** to handle the ~77/23 class imbalance. This prioritizes catching no-shows (recall) over pure accuracy.

### Top 10 Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | Customer Age | 0.1310 |
| 2 | Payment Method | 0.1176 |
| 3 | Past Visit Count | 0.0994 |
| 4 | Booking Lead Time (days) | 0.0970 |
| 5 | Appointment Hour | 0.0934 |
| 6 | Service Type | 0.0803 |
| 7 | Day of Week | 0.0774 |
| 8 | Branch | 0.0715 |
| 9 | Lead Time Bucket | 0.0429 |
| 10 | Hour Bucket | 0.0407 |

---

## 📊 MLflow Experiment Tracking

All model training runs are tracked via **MLflow**:

- **Experiment**: `NoShow_Prediction`
- **Tracked per run**: Model parameters, metrics (AUC-ROC, F1, Precision, Recall, Accuracy), feature importances, confusion matrix, and the serialized sklearn model
- **Model Registry**: Best model registered as `noshow_predictor` with version control
- **UI Access**: `mlflow ui --backend-store-uri file:///path/to/mlruns` → opens at `http://localhost:5000`

### What MLflow Enables

| Capability | Description |
|---|---|
| **Experiment Comparison** | Compare all 4 models side-by-side with logged metrics |
| **Model Versioning** | Track model versions with rollback support |
| **Artifact Storage** | Feature importances, confusion matrices stored per run |
| **Reproducibility** | All hyperparameters logged for exact reproduction |
| **Registry** | Best model registered as `noshow_predictor` for production deployment |

---

## 🔌 FastAPI Prediction API

A working REST API for real-time no-show risk scoring.

```bash
# Start the API server
uvicorn api.api:app --host 0.0.0.0 --port 8000
# Swagger docs: http://localhost:8000/docs
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check + model status |
| `/predict` | POST | Single booking no-show risk prediction |

### Example: Predict No-Show Risk

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_0042",
    "service_type": "Hair Coloring",
    "branch": "Koramangala",
    "booking_lead_time_days": 5,
    "appointment_hour": 14,
    "past_visit_count": 8,
    "past_cancellation_count": 1,
    "past_no_show_count": 2,
    "payment_method": "Online Prepaid",
    "day_of_week": "Wednesday",
    "customer_age": 28,
    "gender": "Female"
  }'
```

**Response:**

```json
{
  "customer_id": "CUST_0042",
  "no_show_probability": 0.309,
  "risk_label": "Medium",
  "risk_color": "#FFA500",
  "recommended_action": "📱 Send reminder SMS 24h + 2h before appointment. Add to waitlist backup queue.",
  "model_version": "Random_Forest_v1"
}
```

---

## 📐 Business Logic & Decision Thresholds

| Predicted Probability | Risk Label | Business Action |
|---|---|---|
| **≥ 60%** | 🔴 High Risk | Require prepaid booking, double SMS reminder, offer reschedule |
| **30% – 59%** | 🟡 Medium Risk | Send reminder 24h + 2h before, waitlist a backup customer |
| **< 30%** | 🟢 Low Risk | Standard confirmation SMS |

### Revenue Impact Estimation

- Each no-show costs the average service price (₹500 – ₹15,000)
- At ~23.5% no-show rate across 10K bookings, estimated revenue loss is **significant**
- Even a **5% reduction** in no-shows can recover substantial monthly revenue

---

## 📈 AI Dashboard

Interactive Streamlit dashboard with **4 tabs**:

### Tab 1: Executive Overview

- KPI cards: Total Bookings, No-Show Rate, Revenue Loss, High-Risk Count
- No-show rate trend over time (weekly)
- High-risk bookings table with probability scores

### Tab 2: AI Insights

- Risk distribution (High/Medium/Low) donut chart
- Feature importance bar chart (top 10)
- No-show rate breakdown by Branch and Service Type
- Model comparison table

### Tab 3: Customer Behavior

- Repeat vs New vs Loyal customer distribution
- Booking lead-time histogram (Show vs No-Show overlay)
- Peak no-show heatmap (Day of Week × Hour)
- No-show rate by Payment Method

### Tab 4: Retention Intelligence

- Customer segmentation pie chart (Loyal/Regular/New/At-Risk/Churned)
- Segment breakdown table with metrics
- Churn-risk customers list with reasons
- 3 data-backed retention strategies

### Filters (Sidebar)

- Date range picker
- Branch multi-select
- Service type multi-select

```bash
# Launch dashboard
streamlit run dashboard/app.py
```

---

## 🔄 Retention Intelligence

### Customer Segments

| Segment | Rule | Description |
|---|---|---|
| **Loyal** | 10+ bookings, <20% no-show ratio | Most valuable, reliable customers |
| **Regular** | 3-9 bookings | Consistently returning customers |
| **New** | 1-2 bookings | Recently acquired, need nurturing |
| **At-Risk** | No visit in 60+ days OR >50% no-show | Showing disengagement signals |
| **Churned** | No visit in 90+ days + had prior visits | Likely lost customers |

### Retention Strategies

1. **🎯 Targeted Re-Engagement Campaign** — Personalized offers to churned/at-risk customers with prepaid booking requirement
2. **💳 Prepaid Booking Incentive** — 10% discount for online prepaid; reduces no-shows by making customers financially committed
3. **⭐ New Customer Onboarding** — "3-visit reward" program to convert new customers to regulars

---

## 🏗 Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                          │
│                  Hair Rap by YoYo — AI Module                       │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   Mobile/Web Client   │
                    │   (Booking Request)   │
                    └──────────┬───────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    API Gateway (Nginx)                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Booking API (FastAPI)                          │
│  ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ POST /book  │  │ GET /risk/{id} │  │ GET /dashboard   │  │
│  └──────┬──────┘  └───────┬────────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
          ▼                 ▼                    │
┌──────────────────────────────────┐             │
│     Feature Engineering          │             │
│     Pipeline (Real-time)         │             │
│  ┌────────────────────────────┐  │             │
│  │ • Customer history lookup  │  │             │
│  │ • Compute no_show_ratio    │  │             │
│  │ • Encode categoricals      │  │             │
│  │ • Time-based features      │  │             │
│  └─────────────┬──────────────┘  │             │
└────────────────┼─────────────────┘             │
                 │                               │
                 ▼                               │
┌──────────────────────────────────┐             │
│    ML Model Service              │             │
│    (Random Forest - Scikit)      │             │
│  ┌────────────────────────────┐  │             │
│  │ predict_proba() → 0.72     │  │             │
│  │ risk_label    → "High"     │  │             │
│  └─────────────┬──────────────┘  │             │
└────────────────┼─────────────────┘             │
                 │                               │
                 ▼                               ▼
┌─────────────────────┐          ┌──────────────────────────┐
│  Business Actions    │         │  Streamlit Dashboard     │
│  ┌────────────────┐  │         │  ┌────────────────────┐  │
│  │ SMS Reminder   │  │         │  │ Executive Overview │  │
│  │ Prepaid Req.   │  │         │  │ AI Insights        │  │
│  │ Overbooking    │  │         │  │ Customer Behavior  │  │
│  │ Waitlist Mgmt  │  │         │  │ Retention Intel.   │  │
│  └────────────────┘  │         │  └────────────────────┘  │
└─────────────────────┘          └──────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│                    PostgreSQL Database                       │
│  ┌──────────┐  ┌────────────┐  ┌────────────────────────┐    │
│  │ bookings │  │ customers  │  │ prediction_logs        │    │
│  └──────────┘  └────────────┘  └────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌──────────────┐ ┌───────────┐ ┌──────────────┐
│ Weekly       │ │ Data Drift│ │ Monitoring   │
│ Retraining   │ │ Detection │ │ (Prometheus) │
│ Pipeline     │ │ (PSI/KS)  │ │              │
│ (Airflow)    │ │           │ │ • AUC-ROC    │
│              │ │ • Feature │ │ • No-show %  │
│ • Fetch new  │ │   distrib.│ │ • Latency    │
│   data       │ │ • Alert   │ │ • Throughput │
│ • Retrain    │ │   if drift│ │              │
│ • Validate   │ │           │ │ Grafana      │
│ • Deploy     │ │           │ │ Dashboards   │
└──────┬───────┘ └───────────┘ └──────────────┘
       │
       ▼
┌──────────────────────┐
│  Model Registry      │
│   (MLflow)           │
│  • Version control   │
│  • A/B testing       │
│  • Rollback support  │
└──────────────────────┘
```

### Model Integration with Booking API

```python
# FastAPI endpoint example
@app.post("/api/v1/predict-risk")
async def predict_risk(booking: BookingRequest):
    features = feature_pipeline.transform(booking)
    probability = model.predict_proba(features)[0][1]
    risk_label = get_risk_label(probability)

    # Trigger business action
    if risk_label == "High":
        await send_reminder(booking.customer_id, priority="urgent")
        await require_prepayment(booking.booking_id)

    return {"risk_score": probability, "risk_label": risk_label}
```

### Real-Time Inference Flow

1. Customer creates booking → Booking API receives request
2. API calls Feature Engineering Pipeline → Fetches customer history from DB, computes 19 features
3. Features sent to ML Model → Returns no-show probability (< 50ms latency)
4. Probability → Risk Label → Business Action triggered automatically
5. Prediction logged to `prediction_logs` table for monitoring

### Retraining Strategy

- **Schedule**: Weekly batch retraining every Sunday at 2 AM
- **Pipeline**: Apache Airflow DAG orchestrates data fetch → retrain → validate → deploy
- **Validation gate**: New model must have AUC-ROC ≥ current model - 0.02 (regression guard)
- **Deployment**: Blue-green deployment — old model stays active until new model passes A/B test
- **Data window**: Rolling 6-month training window to capture seasonal patterns

### Data Drift Detection

- **Method**: Population Stability Index (PSI) + Kolmogorov-Smirnov test
- **Monitored features**: All 19 features, checked daily
- **Threshold**: PSI > 0.2 → alert for investigation; PSI > 0.25 → trigger emergency retrain
- **Distribution tracking**: Daily feature histograms stored for comparison

### Monitoring Metrics

| Metric | Description | Alert Threshold |
|---|---|---|
| AUC-ROC (production) | Weekly evaluated on new data | < 0.55 |
| No-show prediction accuracy | % of correct predictions | < 60% |
| Prediction latency (p99) | 99th percentile response time | > 200ms |
| Daily no-show rate | Actual vs predicted no-show rate | Drift > 5% |
| Model staleness | Days since last retrain | > 14 days |

### Scalability for 200K+ Users

- **Horizontal scaling**: Kubernetes pods with auto-scaling based on prediction request volume
- **Model caching**: Model loaded once in memory per pod, shared across requests via `joblib`
- **Async inference**: FastAPI async endpoints with connection pooling for DB queries
- **Batch prediction**: Nightly batch scoring of next-day bookings (reduces real-time load)
- **Database**: Read replicas for dashboard queries; write primary for booking + prediction logs
- **CDN + Caching**: Dashboard data cached with 5-minute TTL; static assets on CDN
- **Load testing target**: <100ms p95 latency at 1,000 concurrent prediction requests

---

## 📝 Assumptions

1. **Synthetic data**: Dataset is mock-generated with realistic embedded patterns. In production, real booking data would provide stronger signal.
2. **No-show rate ~23.5%**: Based on industry averages for salon/beauty businesses (typically 20-30%).
3. **5 Bangalore branches**: Used as representative locations; scalable to any city/region.
4. **Service prices**: Fixed per service type for revenue estimation; real system would use actual pricing.
5. **Customer history**: Aggregated at booking time (not dynamically updated within the dataset).
6. **Rule-based churn detection**: Acceptable per assignment requirements; production would use a dedicated churn ML model.
7. **Payment method as feature**: Assumes payment method is known at booking time (selected during booking flow).

---

## 🛠 Tech Stack

| Component | Technology | Status |
|---|---|---|
| Language | Python 3.12 | ✅ Implemented |
| ML Models | Scikit-learn, XGBoost, LightGBM | ✅ Implemented |
| Dashboard | Streamlit, Plotly | ✅ Implemented |
| Prediction API | FastAPI + Uvicorn | ✅ Implemented |
| Experiment Tracking | MLflow | ✅ Implemented |
| Model Registry | MLflow Model Registry | ✅ Implemented |
| Data Processing | Pandas, NumPy | ✅ Implemented |
| Visualization | Plotly, Matplotlib, Seaborn | ✅ Implemented |
| Model Serialization | Joblib | ✅ Implemented |
| Orchestration | Apache Airflow | 📐 Designed |
| Monitoring | Prometheus + Grafana | 📐 Designed |
| Database | PostgreSQL | 📐 Designed |
| Container | Docker + Kubernetes | 📐 Designed |

---

**Built by Abhiram Kumar Soni for Hair Rap by YoYo**


