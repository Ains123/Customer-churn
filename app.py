import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('models/churn_model.pkl')

st.title("Customer Churn Prediction")

st.write("Enter customer information below:")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

if st.button("Predict Churn"):

    data = pd.DataFrame({
        "tenure":[tenure],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges],
        "Contract":[contract]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")