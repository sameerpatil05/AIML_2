# app.py
# Customer Purchase Prediction - Streamlit App
# DSBDA Mini Project

import streamlit as st
import numpy as np
import pickle

# Page Title
st.title("🛒 Customer Purchase Predictor")
st.write("Enter customer details below to predict the purchase amount.")

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input Fields
age            = st.number_input("Age", min_value=18, max_value=70, value=30)
annual_income  = st.number_input("Annual Income (₹)", min_value=10000, max_value=200000, value=50000)
spending_score = st.number_input("Spending Score (1 to 100)", min_value=1, max_value=100, value=50)

# Predict Button
if st.button("Predict Purchase Amount"):
    input_data = np.array([[age, annual_income, spending_score]])
    prediction = model.predict(input_data)[0]
    prediction = round(max(30.0, prediction), 2)
    st.success(f"Predicted Purchase Amount: ₹ {prediction}")
