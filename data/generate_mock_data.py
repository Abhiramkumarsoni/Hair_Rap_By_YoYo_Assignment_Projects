"""
Generate synthetic salon booking dataset for Hair Rap by YoYo.
Creates ~10,000 realistic booking records with embedded no-show patterns.
"""


# import necessary library
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# ─── Configuration ────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

NUM_BOOKINGS = 10000
NUM_CUSTOMERS = 2000

BRANCHES = ["Koramangala", "Indiranagar", "Whitefield", "HSR Layout", "Jayanagar"]
BRANCH_WEIGHTS = [0.25, 0.22, 0.20, 0.18, 0.15]  # booking distribution
BRANCH_NO_SHOW_BIAS = {
    "Koramangala": 0.0,      # baseline
    "Indiranagar": -0.02,    # slightly fewer no-shows
    "Whitefield": 0.05,      # higher no-show (farther location)
    "HSR Layout": 0.01,
    "Jayanagar": -0.01,
}

SERVICE_TYPES = ["Haircut", "Hair Coloring", "Spa & Massage", "Facial", "Bridal Package", "Hair Treatment"]
SERVICE_WEIGHTS = [0.30, 0.18, 0.15, 0.15, 0.08, 0.14]
SERVICE_PRICES = {
    "Haircut": 500,
    "Hair Coloring": 2500,
    "Spa & Massage": 3000,
    "Facial": 1500,
    "Bridal Package": 15000,
    "Hair Treatment": 4000,
}
SERVICE_NO_SHOW_BIAS = {
    "Haircut": 0.02,           # cheap → slightly more no-shows
    "Hair Coloring": -0.01,
    "Spa & Massage": -0.02,
    "Facial": 0.01,
    "Bridal Package": -0.08,   # expensive → rarely no-show
    "Hair Treatment": -0.01,
}

PAYMENT_METHODS = ["Pay at Salon (Cash)", "Online Prepaid", "Card on File", "UPI Prepaid"]
PAYMENT_WEIGHTS = [0.35, 0.25, 0.20, 0.20]
PAYMENT_NO_SHOW_BIAS = {
    "Pay at Salon (Cash)": 0.08,   # no skin in the game → higher no-show
    "Online Prepaid": -0.06,       # already paid → very unlikely no-show
    "Card on File": -0.03,
    "UPI Prepaid": -0.05,
}

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.40, 0.55, 0.05]

APPOINTMENT_HOURS = list(range(9, 21))  # 9 AM to 8 PM
HOUR_WEIGHTS = [0.04, 0.08, 0.10, 0.12, 0.10, 0.08, 0.04, 0.06, 0.10, 0.12, 0.10, 0.06]

# Date range: bookings from 6 months ago to today
END_DATE = datetime(2026, 2, 25)
START_DATE = END_DATE - timedelta(days=180)

BASE_NO_SHOW_RATE = 0.20


# ─── Customer Profile Generation ─────────────────────────────────────────────
def generate_customers(n_customers):
    """Generate customer profiles with historical behavior."""
    customers = []
    for cid in range(1, n_customers + 1):
        age = int(np.clip(np.random.normal(32, 10), 18, 65))
        gender = np.random.choice(GENDERS, p=GENDER_WEIGHTS)

        # Some customers are loyal, some are flaky
        loyalty_score = np.random.beta(2, 5)  # skewed toward lower values
        past_visits = int(np.random.exponential(5)) + 1
        past_visits = min(past_visits, 50)

        # Flaky customers have more cancellations and no-shows
        flakiness = np.random.beta(2, 8)
        past_cancellations = int(past_visits * flakiness * np.random.uniform(0.1, 0.5))
        past_no_shows = int(past_visits * flakiness * np.random.uniform(0.1, 0.4))

        customers.append({
            "customer_id": f"CUST_{cid:04d}",
            "customer_age": age,
            "gender": gender,
            "past_visit_count": past_visits,
            "past_cancellation_count": past_cancellations,
            "past_no_show_count": past_no_shows,
            "flakiness": flakiness,
        })

    return pd.DataFrame(customers)


# ─── No-Show Probability Calculation ─────────────────────────────────────────
def calculate_no_show_probability(row, customer_flakiness):
    """
    Calculate no-show probability based on multiple factors.
    This creates realistic, learnable patterns in the data.
    """
    prob = BASE_NO_SHOW_RATE

    # Factor 1: Customer history (strongest signal)
    if row["past_visit_count"] > 0:
        no_show_ratio = row["past_no_show_count"] / (row["past_visit_count"] + 1)
        prob += no_show_ratio * 0.35  # strong positive correlation

    # Factor 2: Customer flakiness (latent trait)
    prob += customer_flakiness * 0.15

    # Factor 3: Booking lead time
    lead_time = row["booking_lead_time_days"]
    if lead_time > 14:
        prob += 0.08
    elif lead_time > 7:
        prob += 0.04
    elif lead_time <= 1:
        prob -= 0.05  # same-day/next-day bookings → committed

    # Factor 4: Payment method
    prob += PAYMENT_NO_SHOW_BIAS[row["payment_method"]]

    # Factor 5: Branch
    prob += BRANCH_NO_SHOW_BIAS[row["branch"]]

    # Factor 6: Service type
    prob += SERVICE_NO_SHOW_BIAS[row["service_type"]]

    # Factor 7: Time of day
    hour = row["appointment_hour"]
    if hour <= 10 or hour >= 19:
        prob += 0.03  # early morning / late evening → more no-shows
    elif 12 <= hour <= 14:
        prob -= 0.02  # lunch-time slots → people are around

    # Factor 8: Day of week
    if row["day_of_week"] in ["Saturday", "Sunday"]:
        prob += 0.02  # weekends slightly higher
    elif row["day_of_week"] == "Monday":
        prob += 0.01  # Monday blues

    # Factor 9: Repeat vs new customer
    if row["past_visit_count"] >= 10:
        prob -= 0.06  # very loyal customers
    elif row["past_visit_count"] <= 1:
        prob += 0.03  # first-timers slightly higher

    # Factor 10: Age effect (slight)
    if row["customer_age"] < 25:
        prob += 0.02  # younger customers slightly more flaky
    elif row["customer_age"] > 45:
        prob -= 0.02  # older customers more reliable

    # Clamp probability
    prob = np.clip(prob, 0.02, 0.85)

    return prob


# ─── Main Data Generation ────────────────────────────────────────────────────
def generate_bookings(n_bookings, customers_df):
    """Generate booking records with realistic patterns."""
    bookings = []

    for i in range(n_bookings):
        # Select a random customer
        customer = customers_df.sample(1).iloc[0]

        # Generate appointment date (random within date range)
        days_offset = np.random.randint(0, (END_DATE - START_DATE).days)
        appointment_date = START_DATE + timedelta(days=days_offset)

        # Generate booking lead time (exponential distribution, most book 1-7 days ahead)
        lead_time = int(np.random.exponential(5)) + 1
        lead_time = min(lead_time, 30)
        booking_date = appointment_date - timedelta(days=lead_time)

        # Select attributes
        branch = np.random.choice(BRANCHES, p=BRANCH_WEIGHTS)
        service = np.random.choice(SERVICE_TYPES, p=SERVICE_WEIGHTS)
        payment = np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
        hour = np.random.choice(APPOINTMENT_HOURS, p=HOUR_WEIGHTS)
        day_name = appointment_date.strftime("%A")

        row = {
            "booking_id": f"BK_{i + 1:05d}",
            "customer_id": customer["customer_id"],
            "service_type": service,
            "service_price": SERVICE_PRICES[service],
            "booking_date": booking_date.strftime("%Y-%m-%d"),
            "appointment_date": appointment_date.strftime("%Y-%m-%d"),
            "appointment_hour": hour,
            "booking_lead_time_days": lead_time,
            "past_visit_count": customer["past_visit_count"],
            "past_cancellation_count": customer["past_cancellation_count"],
            "past_no_show_count": customer["past_no_show_count"],
            "payment_method": payment,
            "branch": branch,
            "day_of_week": day_name,
            "is_weekend": 1 if day_name in ["Saturday", "Sunday"] else 0,
            "customer_age": customer["customer_age"],
            "gender": customer["gender"],
        }

        # Calculate no-show probability and determine outcome
        no_show_prob = calculate_no_show_probability(row, customer["flakiness"])
        row["appointment_outcome"] = "No-Show" if np.random.random() < no_show_prob else "Show"

        bookings.append(row)

    return pd.DataFrame(bookings)


def main():
    print("=" * 60)
    print("  Hair Rap by YoYo — Synthetic Dataset Generator")
    print("=" * 60)

    # Step 1: Generate customer profiles
    print("\n[1/3] Generating customer profiles...")
    customers_df = generate_customers(NUM_CUSTOMERS)
    print(f"  ✓ Created {len(customers_df)} unique customers")

    # Step 2: Generate bookings
    print("\n[2/3] Generating booking records...")
    bookings_df = generate_bookings(NUM_BOOKINGS, customers_df)
    print(f"  ✓ Created {len(bookings_df)} bookings")

    # Step 3: Save to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "salon_bookings.csv")
    bookings_df.to_csv(output_path, index=False)
    print(f"\n[3/3] Saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total bookings:      {len(bookings_df):,}")
    print(f"  Unique customers:    {bookings_df['customer_id'].nunique():,}")
    print(f"  Date range:          {bookings_df['booking_date'].min()} → {bookings_df['appointment_date'].max()}")
    print(f"  Branches:            {bookings_df['branch'].nunique()}")
    print(f"  Service types:       {bookings_df['service_type'].nunique()}")

    outcome_counts = bookings_df["appointment_outcome"].value_counts()
    total = len(bookings_df)
    print(f"\n  Appointment Outcomes:")
    for outcome, count in outcome_counts.items():
        print(f"    {outcome:10s}: {count:,} ({count/total*100:.1f}%)")

    print(f"\n  No-Show Rate by Branch:")
    branch_rates = bookings_df.groupby("branch")["appointment_outcome"].apply(
        lambda x: (x == "No-Show").mean() * 100
    ).sort_values(ascending=False)
    for branch, rate in branch_rates.items():
        print(f"    {branch:15s}: {rate:.1f}%")

    print(f"\n  No-Show Rate by Service:")
    service_rates = bookings_df.groupby("service_type")["appointment_outcome"].apply(
        lambda x: (x == "No-Show").mean() * 100
    ).sort_values(ascending=False)
    for service, rate in service_rates.items():
        print(f"    {service:20s}: {rate:.1f}%")

    print(f"\n  No-Show Rate by Payment Method:")
    payment_rates = bookings_df.groupby("payment_method")["appointment_outcome"].apply(
        lambda x: (x == "No-Show").mean() * 100
    ).sort_values(ascending=False)
    for method, rate in payment_rates.items():
        print(f"    {method:25s}: {rate:.1f}%")

    # Save customer profiles too (useful for retention analysis)
    customers_path = os.path.join(output_dir, "customer_profiles.csv")
    customers_df.drop(columns=["flakiness"]).to_csv(customers_path, index=False)
    print(f"\n  ✓ Customer profiles saved to: {customers_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

