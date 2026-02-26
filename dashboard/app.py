"""
AI-Powered No-Show & Customer Intelligence Dashboard
Hair Rap by YoYo — Enterprise Salon Booking Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import prepare_features, get_feature_names
from src.predict import predict_no_show, get_risk_label, get_risk_color
from src.retention import segment_customers, identify_churn_risk, get_retention_strategies


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hair Rap by YoYo — AI Intelligence",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        font-size: 14px;
        font-weight: 400;
        margin-bottom: 4px;
        opacity: 0.9;
    }
    .metric-card h1 {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .metric-card-red {
        background: linear-gradient(135deg, #f5365c 0%, #f56036 100%);
        box-shadow: 0 4px 15px rgba(245, 54, 92, 0.3);
    }
    .metric-card-green {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);
        box-shadow: 0 4px 15px rgba(45, 206, 137, 0.3);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%);
        box-shadow: 0 4px 15px rgba(251, 99, 64, 0.3);
    }
    .strategy-card {
        background: #f8f9fe;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .header-bar {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Data & Models ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "salon_bookings.csv"))
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
    encoders = joblib.load(os.path.join(PROJECT_ROOT, "models", "label_encoders.pkl"))
    with open(os.path.join(PROJECT_ROOT, "models", "model_metadata.json")) as f:
        metadata = json.load(f)
    fi = pd.read_csv(os.path.join(PROJECT_ROOT, "models", "feature_importance.csv"))
    comparison = pd.read_csv(os.path.join(PROJECT_ROOT, "models", "model_comparison.csv"), index_col=0)
    return model, encoders, metadata, fi, comparison


df = load_data()
model, encoders, metadata, feature_importance, model_comparison = load_model_artifacts()

# Generate predictions for all bookings
X_all, _, _ = prepare_features(df, label_encoders=encoders, fit=False)
predictions = predict_no_show(model, X_all)
df["no_show_probability"] = predictions["no_show_probability"]
df["risk_label"] = predictions["risk_label"]

# Compute customer segments
customer_stats = segment_customers(df)

# Service prices for revenue estimation
SERVICE_PRICES = {
    "Haircut": 500, "Hair Coloring": 2500, "Spa & Massage": 3000,
    "Facial": 1500, "Bridal Package": 15000, "Hair Treatment": 4000,
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💇 Hair Rap by YoYo")
    st.markdown("**AI Intelligence Dashboard**")
    st.markdown("---")

    # Date range filter
    st.markdown("### 📅 Date Range")
    min_date = df["appointment_date"].min().date()
    max_date = df["appointment_date"].max().date()
    date_range = st.date_input(
        "Select Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Branch filter
    st.markdown("### 🏪 Branch")
    all_branches = sorted(df["branch"].unique())
    selected_branches = st.multiselect("Select Branch(es)", all_branches, default=all_branches)

    # Service filter
    st.markdown("### ✂️ Service Type")
    all_services = sorted(df["service_type"].unique())
    selected_services = st.multiselect("Select Service(s)", all_services, default=all_services)

    st.markdown("---")
    st.markdown(f"**Model**: {metadata['best_model_name']}")
    st.markdown(f"**AUC-ROC**: {metadata['metrics']['AUC-ROC']:.4f}")
    st.markdown(f"**Records**: {len(df):,}")


# ─── Apply Filters ────────────────────────────────────────────────────────────
filtered = df.copy()

if len(date_range) == 2:
    start_dt, end_dt = date_range
    filtered = filtered[
        (filtered["appointment_date"].dt.date >= start_dt) &
        (filtered["appointment_date"].dt.date <= end_dt)
    ]

filtered = filtered[filtered["branch"].isin(selected_branches)]
filtered = filtered[filtered["service_type"].isin(selected_services)]


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h1 style="margin:0; font-size:28px;">💇 AI-Powered No-Show & Customer Intelligence</h1>
    <p style="margin:4px 0 0 0; opacity:0.8;">Enterprise salon booking intelligence • Hair Rap by YoYo</p>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🤖 AI Insights",
    "👥 Customer Behavior",
    "🔄 Retention Intelligence",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Executive Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Executive Overview")

    # KPI Metrics
    total_bookings = len(filtered)
    no_shows = len(filtered[filtered["appointment_outcome"] == "No-Show"])
    no_show_rate = no_shows / total_bookings * 100 if total_bookings > 0 else 0
    high_risk = len(filtered[filtered["risk_label"] == "High"])

    # Revenue impact
    no_show_bookings = filtered[filtered["appointment_outcome"] == "No-Show"]
    revenue_loss = no_show_bookings["service_price"].sum() if "service_price" in no_show_bookings.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Bookings</h3>
            <h1>{total_bookings:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-red">
            <h3>No-Show Rate</h3>
            <h1>{no_show_rate:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card metric-card-orange">
            <h3>Est. Revenue Loss</h3>
            <h1>₹{revenue_loss:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <h3>High-Risk Bookings</h3>
            <h1>{high_risk:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # No-show trend over time
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📈 No-Show Rate Trend Over Time")
        trend = filtered.groupby(filtered["appointment_date"].dt.to_period("W")).agg(
            total=("appointment_outcome", "count"),
            no_shows=("appointment_outcome", lambda x: (x == "No-Show").sum()),
        ).reset_index()
        trend["appointment_date"] = trend["appointment_date"].dt.to_timestamp()
        trend["no_show_rate"] = trend["no_shows"] / trend["total"] * 100

        fig_trend = px.area(
            trend, x="appointment_date", y="no_show_rate",
            labels={"appointment_date": "Week", "no_show_rate": "No-Show Rate (%)"},
            color_discrete_sequence=["#667eea"],
        )
        fig_trend.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_trend.update_traces(fill="tozeroy", fillcolor="rgba(102, 126, 234, 0.15)")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Outcome Distribution")
        outcome_counts = filtered["appointment_outcome"].value_counts()
        fig_pie = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            color=outcome_counts.index,
            color_discrete_map={"Show": "#2dce89", "No-Show": "#f5365c"},
            hole=0.5,
        )
        fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        fig_pie.update_traces(textinfo="label+percent", textfont_size=14)
        st.plotly_chart(fig_pie, use_container_width=True)

    # High-risk upcoming bookings table
    st.markdown("#### ⚠️ High-Risk Bookings")
    high_risk_df = filtered[filtered["risk_label"] == "High"].sort_values(
        "no_show_probability", ascending=False
    ).head(15)[["booking_id", "customer_id", "appointment_date", "branch",
                 "service_type", "payment_method", "no_show_probability", "risk_label"]]

    if len(high_risk_df) > 0:
        high_risk_df["no_show_probability"] = high_risk_df["no_show_probability"].apply(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)
    else:
        st.info("No high-risk bookings found for the selected filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: AI Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### AI Model Insights")

    col_left, col_right = st.columns(2)

    with col_left:
        # Risk distribution
        st.markdown("#### 🎯 Risk Distribution")
        risk_counts = filtered["risk_label"].value_counts()
        colors_map = {"High": "#f5365c", "Medium": "#fb6340", "Low": "#2dce89"}
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=colors_map,
            hole=0.45,
        )
        fig_risk.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        fig_risk.update_traces(textinfo="label+value+percent", textfont_size=13)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        # Feature importance
        st.markdown("#### 📊 Feature Importance (Top 10)")
        fi_top = feature_importance.head(10).sort_values("importance", ascending=True)
        fig_fi = px.bar(
            fi_top, x="importance", y="feature_display",
            orientation="h",
            color="importance",
            color_continuous_scale=["#667eea", "#764ba2"],
            labels={"importance": "Importance", "feature_display": ""},
        )
        fig_fi.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        # No-show rate by branch
        st.markdown("#### 🏪 No-Show Rate by Branch")
        branch_rates = filtered.groupby("branch")["appointment_outcome"].apply(
            lambda x: (x == "No-Show").mean() * 100
        ).sort_values(ascending=False).reset_index()
        branch_rates.columns = ["Branch", "No-Show Rate (%)"]

        fig_branch = px.bar(
            branch_rates, x="Branch", y="No-Show Rate (%)",
            color="No-Show Rate (%)",
            color_continuous_scale=["#2dce89", "#fb6340", "#f5365c"],
            text="No-Show Rate (%)",
        )
        fig_branch.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_branch.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_branch, use_container_width=True)

    with col_right2:
        # No-show rate by service
        st.markdown("#### ✂️ No-Show Rate by Service")
        service_rates = filtered.groupby("service_type")["appointment_outcome"].apply(
            lambda x: (x == "No-Show").mean() * 100
        ).sort_values(ascending=False).reset_index()
        service_rates.columns = ["Service", "No-Show Rate (%)"]

        fig_service = px.bar(
            service_rates, x="Service", y="No-Show Rate (%)",
            color="No-Show Rate (%)",
            color_continuous_scale=["#2dce89", "#fb6340", "#f5365c"],
            text="No-Show Rate (%)",
        )
        fig_service.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_service.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_service, use_container_width=True)

    # Model comparison table
    st.markdown("#### 🏆 Model Comparison")
    styled_comparison = model_comparison.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda")
    st.dataframe(styled_comparison, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Customer Behavior
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Customer Behavior Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        # Repeat vs New customers
        st.markdown("#### 👤 Repeat vs New Customers")
        filtered_with_type = filtered.copy()
        filtered_with_type["customer_type"] = filtered_with_type["past_visit_count"].apply(
            lambda x: "New (1-2 visits)" if x <= 2 else ("Regular (3-9)" if x < 10 else "Loyal (10+)")
        )
        cust_type_counts = filtered_with_type["customer_type"].value_counts()
        fig_cust = px.pie(
            values=cust_type_counts.values,
            names=cust_type_counts.index,
            color_discrete_sequence=["#667eea", "#2dce89", "#fbb140"],
            hole=0.45,
        )
        fig_cust.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        fig_cust.update_traces(textinfo="label+value+percent", textfont_size=12)
        st.plotly_chart(fig_cust, use_container_width=True)

    with col_right:
        # Booking lead-time analysis
        st.markdown("#### ⏳ Booking Lead-Time Distribution")
        fig_lead = px.histogram(
            filtered, x="booking_lead_time_days", color="appointment_outcome",
            barmode="overlay", nbins=30,
            color_discrete_map={"Show": "#2dce89", "No-Show": "#f5365c"},
            labels={"booking_lead_time_days": "Lead Time (Days)", "appointment_outcome": "Outcome"},
            opacity=0.7,
        )
        fig_lead.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_lead, use_container_width=True)

    # Peak no-show time slots heatmap
    st.markdown("#### 🔥 Peak No-Show Heatmap (Day × Hour)")
    heatmap_data = filtered[filtered["appointment_outcome"] == "No-Show"].groupby(
        ["day_of_week", "appointment_hour"]
    ).size().reset_index(name="no_show_count")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_pivot = heatmap_data.pivot_table(
        index="day_of_week", columns="appointment_hour",
        values="no_show_count", fill_value=0,
    ).reindex(day_order)

    fig_heat = px.imshow(
        heatmap_pivot,
        labels=dict(x="Appointment Hour", y="Day of Week", color="No-Shows"),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
    )
    fig_heat.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    # No-show rate by customer type
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.markdown("#### 📊 No-Show Rate by Customer Type")
        cust_noshow = filtered_with_type.groupby("customer_type")["appointment_outcome"].apply(
            lambda x: (x == "No-Show").mean() * 100
        ).reset_index()
        cust_noshow.columns = ["Customer Type", "No-Show Rate (%)"]
        fig_cust_ns = px.bar(
            cust_noshow, x="Customer Type", y="No-Show Rate (%)",
            color="No-Show Rate (%)",
            color_continuous_scale=["#2dce89", "#fb6340", "#f5365c"],
            text="No-Show Rate (%)",
        )
        fig_cust_ns.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cust_ns.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cust_ns, use_container_width=True)

    with col_right2:
        st.markdown("#### 💳 No-Show Rate by Payment Method")
        pay_noshow = filtered.groupby("payment_method")["appointment_outcome"].apply(
            lambda x: (x == "No-Show").mean() * 100
        ).sort_values(ascending=False).reset_index()
        pay_noshow.columns = ["Payment Method", "No-Show Rate (%)"]
        fig_pay = px.bar(
            pay_noshow, x="Payment Method", y="No-Show Rate (%)",
            color="No-Show Rate (%)",
            color_continuous_scale=["#2dce89", "#fb6340", "#f5365c"],
            text="No-Show Rate (%)",
        )
        fig_pay.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_pay.update_layout(
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pay, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Retention Intelligence
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Retention Intelligence")

    # Customer segments
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🎯 Customer Segmentation")
        seg_counts = customer_stats["segment"].value_counts()
        seg_colors = {
            "Loyal": "#2dce89", "Regular": "#667eea",
            "New": "#fbb140", "At-Risk": "#fb6340", "Churned": "#f5365c"
        }
        fig_seg = px.pie(
            values=seg_counts.values,
            names=seg_counts.index,
            color=seg_counts.index,
            color_discrete_map=seg_colors,
            hole=0.45,
        )
        fig_seg.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        fig_seg.update_traces(textinfo="label+value+percent", textfont_size=12)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Segment Breakdown")
        seg_summary = customer_stats.groupby("segment").agg(
            count=("customer_id", "count"),
            avg_bookings=("total_bookings", "mean"),
            avg_no_show_ratio=("no_show_ratio", "mean"),
            avg_days_since_visit=("days_since_last_visit", "mean"),
        ).round(2)
        seg_summary.columns = ["Customers", "Avg Bookings", "Avg No-Show Ratio", "Avg Days Since Visit"]
        st.dataframe(seg_summary, use_container_width=True)

        # Segment KPIs
        total_cust = len(customer_stats)
        loyal_pct = len(customer_stats[customer_stats["segment"] == "Loyal"]) / total_cust * 100
        churn_pct = len(customer_stats[customer_stats["segment"].isin(["At-Risk", "Churned"])]) / total_cust * 100

        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric("Loyal Customers", f"{loyal_pct:.1f}%")
        with kpi_col2:
            st.metric("At-Risk / Churned", f"{churn_pct:.1f}%", delta=f"-{churn_pct:.1f}%", delta_color="inverse")

    st.markdown("---")

    # Churn-risk customers
    st.markdown("#### ⚠️ Churn-Risk Customers (Top 20)")
    churn_risk = identify_churn_risk(customer_stats)
    if len(churn_risk) > 0:
        display_churn = churn_risk.head(20)[[
            "customer_id", "segment", "total_bookings", "total_no_shows",
            "no_show_ratio", "days_since_last_visit", "risk_reasons"
        ]].copy()
        display_churn["no_show_ratio"] = display_churn["no_show_ratio"].apply(lambda x: f"{x:.0%}")
        st.dataframe(display_churn, use_container_width=True, hide_index=True)
    else:
        st.success("No churn-risk customers identified!")

    st.markdown("---")

    # Retention strategies
    st.markdown("#### 💡 Data-Backed Retention Strategies")
    strategies = get_retention_strategies(customer_stats)

    for s in strategies:
        st.markdown(f"""
        <div class="strategy-card">
            <h4 style="margin:0 0 8px 0;">{s['title']}</h4>
            <p style="margin:0 0 8px 0;">{s['description']}</p>
            <p style="margin:0; font-size:13px;">
                <b>Expected Impact:</b> {s['impact']} &nbsp;|&nbsp;
                <b>Effort:</b> {s['effort']} &nbsp;|&nbsp;
                <b>Data Backing:</b> {s['data_backing']}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.5; font-size:13px;'>"
    "💇 Hair Rap by YoYo — AI-Powered Salon Intelligence Dashboard • Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True,
)
