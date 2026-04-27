# app.py — Customer Purchase Amount Predictor
# Run: streamlit run app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Purchase Predictor", page_icon="🛒")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

model = load_model()
df    = load_data()

# ── Title ──────────────────────────────────────────────────────
st.title("🛒 Customer Purchase Amount Predictor")
st.write("Enter customer details to predict the **Purchase Amount in INR**.")
st.divider()

# ── Inputs ─────────────────────────────────────────────────────
age      = st.number_input("🧑 Age",                       min_value=18,  max_value=65,   value=30,   step=1)
income   = st.number_input("💰 Annual Income (in LPA)",    min_value=2.0, max_value=25.0, value=8.0,  step=0.5)
time_web = st.number_input("🌐 Time on Website (minutes)", min_value=5.0, max_value=60.0, value=25.0, step=1.0)
products = st.number_input("📦 Products Browsed",          min_value=1,   max_value=30,   value=15,   step=1)
discount = st.number_input("🏷️ Discount Availed (%)",      min_value=0.0, max_value=30.0, value=10.0, step=0.5)
st.divider()

# ── Predict ────────────────────────────────────────────────────
if st.button("🔮 Predict Purchase Amount", use_container_width=True):
    input_df = pd.DataFrame(
        [[age, income, time_web, products, discount]],
        columns=["Age","Annual_Income_LPA","Time_On_Website",
                 "Products_Browsed","Discount_Availed"]
    )
    prediction = float(np.clip(model.predict(input_df)[0], 500, 25000))

    st.success(f"### 💳 Predicted Purchase Amount: ₹{prediction:,.0f}")

    if prediction >= 15000:
        category = "🔥 High Value"
    elif prediction >= 8000:
        category = "✅ Medium Value"
    else:
        category = "💡 Low Value"

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer Segment", category)
    col2.metric("Purchase Amount",  f"₹{prediction:,.0f}")
    col3.metric("Est. GST (18%)",   f"₹{prediction * 0.18:,.0f}")

# ── Model Metrics ──────────────────────────────────────────────
with st.expander("📈 Model Metrics"):
    st.write("**Test Set Evaluation (20% data)**")
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE",  "₹401")
    col2.metric("MSE",  "₹2,56,491")
    col3.metric("RMSE", "₹506")
    col4.metric("R²",   "97.20%")
    col5.metric("MAPE", "3.25%")

# ── Dataset ────────────────────────────────────────────────────
with st.expander("🗂️ View Dataset"):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

st.caption("Built with Python · Scikit-learn · Streamlit")
