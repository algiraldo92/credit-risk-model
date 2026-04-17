import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# config
st.set_page_config(page_title="Credit Risk App", layout="centered")

# cargar modelo
model = joblib.load("modelo.pkl")

# título
st.title("Credit Risk Scoring Tool")
st.markdown("Predict the probability of customer default based on financial behavior.")

st.write("Enter customer information:")

# inputs
customer_age = st.number_input("Customer Age", 18, 100)
monthly_income = st.number_input("Monthly Income ($)", 0, 20000, step=100)
total_debt = st.number_input("Total Debt ($)", 0, 20000, step=100)
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

    # 1. Probabilidad
    st.subheader(f"Default Probability: {prob*100:.1f}%")

    # 2. Barra visual
    st.progress(float(prob))

    # 3. Clasificación
    if prob > 0.7:
        st.error("🔴 High Risk")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # 4. Métrica 
    st.metric(
        label="Risk Score",
        value=f"{prob*100:.1f}%",
        delta="High Risk" if prob > 0.7 else "Medium Risk" if prob > 0.4 else "Low Risk"
    )

    # 5. Gráfico perfil cliente
    st.subheader("Customer Financial Profile")

    # Variables financieras (dinero)
    financial_features = {
        "Income": monthly_income,
        "Debt": total_debt
    }
    
    fig1, ax1 = plt.subplots(figsize=(5,3))
    ax1.bar(financial_features.keys(), financial_features.values())
    ax1.set_title("Financial Overview ($)")
    ax1.set_ylabel("Amount ($)")

    #2. Variables de comportamiento (conteos)
    behavior_features = {
        "Loans": active_loans_count,
        "Delinquencies": delinquency_count
    }

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(behavior_features.keys(), behavior_features.values())
    ax2.set_title("Credit Behavior")
    ax2.set_ylabel("Count")

    # Mostrar lado a lado (dashboard style)
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig1)

    with col2:
        st.pyplot(fig2)

    # 6. Gauge simple de riesgo
    st.subheader("Risk Visualization")

    fig2, ax2 = plt.subplots()
    ax2.barh(
        ["Risk"],
        [prob],
        color="red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
    )
    ax2.set_xlim(0, 1)

    st.pyplot(fig2)

    # 7. Explicación 
    st.write(
        "This score estimates the likelihood that a customer will default "
        "based on financial behavior patterns."
    )