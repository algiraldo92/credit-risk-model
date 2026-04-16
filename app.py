import streamlit as st
import joblib
import pandas as pd

"""
========================================
APP: CREDIT RISK PREDICTION
========================================

Aplicación interactiva para predecir
la probabilidad de incumplimiento de un cliente.
"""

# cargar modelo
model = joblib.load("modelo.pkl")

st.title("Credit Risk Prediction App")

st.write("Enter customer information:")

# inputs
customer_age = st.number_input("Customer Age", 18, 100)
monthly_income = st.number_input("Monthly Income", 0, 20000)
total_debt = st.number_input("Total Debt", 0, 20000)
active_loans_count = st.number_input("Active Loans Count", 0, 20)
delinquency_count = st.number_input("Delinquency Count", 0, 10)

employment_type = st.selectbox(
    "Employment Type",
    ["formal", "informal", "self_employed"]
)

# predicción
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "customer_age": customer_age,
        "monthly_income": monthly_income,
        "total_debt": total_debt,
        "active_loans_count": active_loans_count,
        "delinquency_count": delinquency_count,
        "employment_type": employment_type
    }])

    prob = model.predict_proba(input_data)[0, 1]

    st.subheader(f"Default Probability: {prob:.2f}")

    # interpretación
    if prob > 0.7:
        st.error("🔴 High Risk")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")