import streamlit as st
import requests
import pandas as pd

# Set the FastAPI endpoint URL
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="No-Show Predictor (API Frontend)",
    page_icon="💇‍♀️",
    layout="centered"
)

# Custom minimal styling
st.markdown("""
<style>
    .risk-high { color: #f5365c; font-weight: bold; }
    .risk-medium { color: #fb6340; font-weight: bold; }
    .risk-low { color: #2dce89; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("💇‍♀️ Predict via Fast API")
st.markdown("Enter booking details below to predict the likelihood of a no-show using the REST API (`http://localhost:8000`).")

with st.form("prediction_form"):
    st.subheader("Booking Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.text_input("Customer ID", "CUST_9999")
        service_type = st.selectbox("Service Type", [
            "Haircut", "Hair Coloring", "Spa & Massage", 
            "Facial", "Bridal Package", "Hair Treatment"
        ])
        branch = st.selectbox("Branch", [
            "Koramangala", "Indiranagar", "Jayanagar", 
            "Whitefield", "HSR Layout", "Malleshwaram", "JP Nagar"
        ])
        appointment_hour = st.slider("Appointment Hour (24h)", 9, 20, 14, help="9 AM to 8 PM")
        day_of_week = st.selectbox("Day of Week", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        
    with col2:
        customer_age = st.number_input("Customer Age", 18, 80, 28)
        booking_lead_time_days = st.number_input("Booking Lead Time (Days)", 0, 365, 5)
        past_visit_count = st.number_input("Past Visits", 0, 100, 2)
        past_cancellation_count = st.number_input("Past Cancellations", 0, 50, 0)
        past_no_show_count = st.number_input("Past No-Shows", 0, 50, 0)
        payment_method = st.selectbox("Payment Method", ["Online Prepaid", "Pay at Salon", "Card"])
        
    submit_button = st.form_submit_button("Get Prediction 🚀")

if submit_button:
    # Prepare payload exactly as expected by the BaseModel in api/api.py
    payload = {
        "customer_id": customer_id,
        "service_type": service_type,
        "branch": branch,
        "booking_lead_time_days": booking_lead_time_days,
        "appointment_hour": appointment_hour,
        "past_visit_count": past_visit_count,
        "past_cancellation_count": past_cancellation_count,
        "past_no_show_count": past_no_show_count,
        "payment_method": payment_method,
        "day_of_week": day_of_week,
        "customer_age": customer_age,
        "gender": gender
    }
    
    with st.spinner("Calling FastAPI..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.success("API Call Successful!")
                
                # Display Results
                st.markdown("### 🎯 API Response")
                
                # Format visually
                risk = result['risk_label']
                color_class = "risk-low" if risk == "Low" else "risk-medium" if risk == "Medium" else "risk-high"
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Probability of No-Show", f"{result['no_show_probability']:.1%}")
                with res_col2:
                    st.markdown(f"**Risk Level:** <span class='{color_class}'>{risk}</span>", unsafe_allow_html=True)
                    
                st.info(f"**Recommended Action:**\n\n{result['recommended_action']}")
            else:
                st.error(f"API returned an error (Status {response.status_code}):\n{response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Is FastAPI running? (Try running `python api/api.py` first)")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
