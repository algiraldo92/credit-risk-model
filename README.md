# Credit Risk Prediction App
## Overview

This project is an end-to-end Machine Learning application that predicts the probability of a customer defaulting on their financial obligations.

It simulates a real-world credit risk use case, including:

Data generation
Feature engineering
Model training
Deployment via an interactive web app
🎯 Business Problem

Financial institutions need to assess the risk of lending money to customers.

This project aims to:

Predict whether a customer is likely to default (fail to pay) based on their financial profile.

## Dataset

The dataset is synthetically generated to simulate real-world credit data.

 Features
Variable	Description
customer_age	Age of the customer
monthly_income	Monthly income (proxy for repayment capacity)
total_debt	Total outstanding debt
active_loans_count	Number of active loans
delinquency_count	Number of past missed payments
employment_type	Type of employment (formal, informal, self-employed)
 Target
Variable	Description
default_flag	1 = Default, 0 = No default
⚙️ Model
Algorithm: Logistic Regression
Pipeline:
Numerical scaling → StandardScaler
Categorical encoding → OneHotEncoder
Evaluation metric:
ROC-AUC

## Results

The model achieves a strong ability to distinguish between risky and non-risky customers.

Example output:

ROC-AUC: ~0.80+
🧱 Project Structure
project/
│── data.py          # Data simulation
│── train.py         # Model training
│── app.py           # Streamlit application
│── modelo.pkl       # Trained model
│── requirements.txt # Dependencies
│── README.md
▶️ How to Run Locally
1. Generate data
python data.py
2. Train model
python train.py
3. Run the app
streamlit run app.py

## Deployment

The application can be deployed easily using Streamlit Cloud:

Push the project to GitHub
Go to Streamlit Cloud
Select your repository
Deploy app.py
🖥️ App Features
User-friendly interface
Real-time probability prediction
Risk classification:
🟢 Low Risk
🟡 Medium Risk
🔴 High Risk

Key Learnings
End-to-end ML workflow
Data preprocessing with pipelines
Model evaluation using ROC-AUC
Deployment of ML models using Streamlit


## Author

Alan Giraldo

## Notes
Dataset is simulated for educational purposes
Inspired by real-world credit risk modeling problems
