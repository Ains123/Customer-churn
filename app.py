import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained pipeline
# -----------------------------
model = joblib.load("models/best_model.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction App")

st.markdown("""
Predict whether a telecom customer is likely to churn based on their account information and usage patterns.
""")

st.divider()

# -----------------------------
# Customer Information
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    account_length = st.number_input(
        "Account Length",
        min_value=1,
        max_value=300,
        value=100
    )

    state = st.text_input(
        "State",
        value="OH"
    )

    area_code = st.selectbox(
        "Area Code",
        [408, 415, 510]
    )

    international_plan = st.selectbox(
        "International Plan",
        ["Yes", "No"]
    )

    voice_mail_plan = st.selectbox(
        "Voice Mail Plan",
        ["Yes", "No"]
    )

    number_vmail_messages = st.number_input(
        "Number of Voicemail Messages",
        min_value=0,
        max_value=60,
        value=0
    )

    customer_service_calls = st.number_input(
        "Customer Service Calls",
        min_value=0,
        max_value=10,
        value=1
    )

with col2:

    total_day_minutes = st.number_input(
        "Total Day Minutes",
        value=180.0
    )

    total_day_calls = st.number_input(
        "Total Day Calls",
        value=100
    )

    total_day_charge = st.number_input(
        "Total Day Charge",
        value=30.60
    )

    total_eve_minutes = st.number_input(
        "Total Evening Minutes",
        value=200.0
    )

    total_eve_calls = st.number_input(
        "Total Evening Calls",
        value=100
    )

    total_eve_charge = st.number_input(
        "Total Evening Charge",
        value=17.00
    )

    total_night_minutes = st.number_input(
        "Total Night Minutes",
        value=200.0
    )

    total_night_calls = st.number_input(
        "Total Night Calls",
        value=100
    )

    total_night_charge = st.number_input(
        "Total Night Charge",
        value=9.00
    )

    total_intl_minutes = st.number_input(
        "Total International Minutes",
        value=10.0
    )

    total_intl_calls = st.number_input(
        "Total International Calls",
        value=4
    )

    total_intl_charge = st.number_input(
        "Total International Charge",
        value=2.70
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn", use_container_width=True):

    data = pd.DataFrame({

        "account length":[account_length],
        "state":[state],
        "area code":[area_code],
        "international plan":[international_plan],
        "voice mail plan":[voice_mail_plan],
        "number vmail messages":[number_vmail_messages],
        "total day minutes":[total_day_minutes],
        "total day calls":[total_day_calls],
        "total day charge":[total_day_charge],
        "total eve minutes":[total_eve_minutes],
        "total eve calls":[total_eve_calls],
        "total eve charge":[total_eve_charge],
        "total night minutes":[total_night_minutes],
        "total night calls":[total_night_calls],
        "total night charge":[total_night_charge],
        "total intl minutes":[total_intl_minutes],
        "total intl calls":[total_intl_calls],
        "total intl charge":[total_intl_charge],
        "customer service calls":[customer_service_calls]

    })

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    churn_probability = probability[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2%}"
    )

    st.progress(float(churn_probability))

    with st.expander("View Input Data"):
        st.subheader("Customer Information")
        st.dataframe(data, use_container_width=True)
    
    st.metric("Churn Probability", f"{churn_probability:.1%}")

    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to churn.")

    st.markdown("""
    ### Recommended Retention Actions
    - Contact the customer with a personalized retention offer.
    - Review recent customer service interactions.
    - Offer a loyalty discount or upgraded plan.
    """)
else:
    st.success("✅ This customer is likely to stay.")

    st.markdown("""
    ### Recommended Action
    Continue providing quality service and consider offering premium products.
    """)

    st.sidebar.title("About")

st.sidebar.info(
"""
This application predicts whether a telecom customer is likely to churn using a machine learning model trained on historical customer data.

**Model:** Random Forest

**Dataset:** Orange Telecom Customer Churn
"""
)

st.markdown("---")
st.caption("Built by Ainsley Nyambura Gichimu | Customer Churn Prediction Project")

