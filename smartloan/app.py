import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from recommendation_engine import generate_recommendations

# Load model
model = joblib.load("model.pkl")

# Page styling
st.set_page_config(page_title="Smart Loan Predictor", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌾 Smart Loan Accessibility Predictor for Farmers")

# Input form
st.header("Enter Farmer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    land_size = st.number_input("Land Size (in acres)", min_value=0.5, max_value=20.0, value=2.0)
    crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Sugarcane", "Maize", "Cotton"])
    dependents = st.slider("Dependents", 0, 10, 2)

with col2:
    income = st.number_input("Annual Income (₹)", min_value=10000, max_value=500000, value=100000)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    prev_loan = st.selectbox("Previous Loan", ["Yes", "No"])
    current_debt = st.number_input("Current Debt (₹)", min_value=0, max_value=200000, value=20000)
    region = st.selectbox("Region", ["Karnataka", "Punjab", "Bihar", "Maharashtra", "Tamil Nadu"])
    subsidy = st.selectbox("Subsidy Access", ["Yes", "No"])

# Submit button
if st.button("Predict Loan Eligibility"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Land_Size": land_size,
        "Crop_Type": {"Wheat":0, "Rice":1, "Sugarcane":2, "Maize":3, "Cotton":4}[crop_type],
        "Annual_Income": income,
        "Credit_Score": credit_score,
        "Previous_Loan": 1 if prev_loan == "Yes" else 0,
        "Current_Debt": current_debt,
        "Region": {"Karnataka":0, "Punjab":1, "Bihar":2, "Maharashtra":3, "Tamil Nadu":4}[region],
        "Subsidy_Access": 1 if subsidy == "Yes" else 0,
        "Dependents": dependents
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.subheader("Prediction Result:")
    st.success(result) if prediction == 1 else st.error(result)

    # SHAP explanation
    explainer = shap.Explainer(model.predict, input_data)
    shap_values = explainer(input_data)
    st.subheader("Feature Impact (SHAP)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt)

    # Recommendations
    if prediction == 0:
        st.subheader("Recommendations to Improve Eligibility:")
        raw_input = {
            "Credit_Score": credit_score,
            "Annual_Income": income,
            "Current_Debt": current_debt,
            "Land_Size": land_size,
            "Subsidy_Access": subsidy
        }
        tips = generate_recommendations(raw_input)
        for tip in tips:
            st.info(tip)

# Fairness dashboard
st.markdown("---")
st.header("📊 Fairness Dashboard")
st.image("fairness_gender.png", caption="Fairness Metrics by Gender", use_column_width=True)
st.markdown("""
This chart shows how your model performs across gender groups. 
If one group has lower accuracy or higher false positives, it may indicate bias.
""")
