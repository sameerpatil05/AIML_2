"""
app.py – Streamlit UI for Customer Purchase Prediction
DSBDA Mini Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Purchase Predictor",
    page_icon="🛒",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

artefact = load_model()
model  = artefact["model"]
scaler = artefact["scaler"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#4C72B0;'>🛒 Customer Purchase Predictor</h1>
    <p style='text-align:center; color:#555; font-size:16px;'>
        DSBDA Mini Project &nbsp;|&nbsp; Linear Regression Model
    </p>
    <hr style='border:1px solid #e0e0e0;'>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar – About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
    st.markdown("### About")
    st.info(
        "This app predicts the **Purchase Amount (₹)** a customer is likely "
        "to spend based on their profile. Built with a Linear Regression model "
        "trained on 200 synthetic customer records."
    )
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("R² Score", "0.8831")
    st.metric("RMSE", "₹20.71")
    st.metric("MAE",  "₹16.41")
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & scikit-learn")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Age (years)",
        min_value=18, max_value=100, value=30, step=1,
        help="Customer age in years",
    )
    annual_income = st.number_input(
        "💰 Annual Income (₹)",
        min_value=10000, max_value=500000, value=60000, step=1000,
        help="Customer's yearly income",
    )
    spending_score = st.number_input(
        "📊 Spending Score (1–100)",
        min_value=1, max_value=100, value=50, step=1,
        help="Score assigned by the store based on spending behaviour",
    )

with col2:
    years_as_customer = st.number_input(
        "📅 Years as Customer",
        min_value=1, max_value=30, value=5, step=1,
        help="How long this person has been a customer",
    )
    number_of_purchases = st.number_input(
        "🛍️ Number of Purchases",
        min_value=1, max_value=200, value=15, step=1,
        help="Total purchases made so far",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Purchase Amount", use_container_width=True, type="primary"):
    input_data = np.array([[age, annual_income, spending_score,
                             years_as_customer, number_of_purchases]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    prediction   = max(50.0, round(prediction, 2))

    st.markdown("---")
    st.success(f"💸 Predicted Purchase Amount: **₹ {prediction:,.2f}**")

    # Extra context
    if prediction < 100:
        tag, color = "Low Spender 🟡", "#FFA500"
    elif prediction < 200:
        tag, color = "Medium Spender 🟢", "#28A745"
    else:
        tag, color = "High Spender 🔴", "#DC3545"

    st.markdown(
        f"<div style='text-align:center; font-size:18px; color:{color}; "
        f"font-weight:bold; margin-top:8px;'>Customer Segment: {tag}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📄 Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Age", "Annual Income (₹)", "Spending Score",
                        "Years as Customer", "Number of Purchases"],
            "Value":   [age, annual_income, spending_score,
                        years_as_customer, number_of_purchases],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ── EDA plots tab ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 View EDA Plots"):
    if os.path.exists("eda_plots.png"):
        st.image("eda_plots.png", caption="EDA – Feature vs Purchase Amount", use_container_width=True)
    if os.path.exists("actual_vs_predicted.png"):
        st.image("actual_vs_predicted.png", caption="Actual vs Predicted", use_container_width=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
with st.expander("📂 View Dataset Sample"):
    df = pd.read_csv("dataset.csv")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
