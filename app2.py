import streamlit as st
import pandas as pd
import joblib

# -------------------- LOAD MODEL --------------------
# Load trained model and label encoder together
# model ,LE = joblib.load("credit_risk_model.joblib")
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
categorical_encoders = joblib.load("categorical_encoders.pkl")
# -------------------- APP HEADER --------------------
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")
st.title("📊 Loan Application Risk Prediction App")
st.write("Fill in the details below to predict whether a loan application is **Poor, Average, or Good**.")

# -------------------- USER INPUT FORM --------------------
with st.form("loan_form"):

    st.subheader("Personal Information")
    # cust_id = st.text_input("Customer ID")
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Gender", ["M", "F"])
    marital_status = st.selectbox("Marital Status", ["Married","Single"])
    employment_status = st.selectbox("Employment Status", ["Salaried","Self-Employed"])
    income = st.number_input("Annual Income (₹)", min_value=0, step=1000)
    number_of_dependants = st.number_input("Number of Dependants", min_value=0, step=1)
    residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgaged"])
    years_at_current_address = st.number_input("Years at Current Address", min_value=0, step=1)
    city = st.selectbox("City", ["Delhi", "Chennai", "Kolkata", "Bangalore", "Pune",
    "Jaipur", "Lucknow", "Mumbai", "Ahmedabad", "Hyderabad"])
    state = st.selectbox("State", ["Delhi", "Tamil Nadu", "West Bengal", "Karnataka", "Maharashtra",
    "Rajasthan", "Uttar Pradesh", "Gujarat", "Telangana"])
    # city = st.text_input("City")
    # state = st.text_input("State")
    zipcode = st.text_input("Zipcode")

    st.subheader("Loan Information")
    # loan_id = st.text_input("Loan ID")
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Personal", "Business", "Other"])
    loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])
    sanction_amount = st.number_input("Sanction Amount", min_value=0, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
    processing_fee = st.number_input("Processing Fee", min_value=0, step=100)
    gst = st.number_input("GST", min_value=0, step=100)
    net_disbursement = st.number_input("Net Disbursement", min_value=0, step=1000)
    loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, max_value=480, step=1)
    principal_outstanding = st.number_input("Principal Outstanding", min_value=0, step=1000)
    bank_balance_at_application = st.number_input("Bank Balance at Application", min_value=0, step=1000)

    # st.subheader("Dates")
    # disbursal_date = st.date_input("Disbursal Date")
    # installment_start_dt = st.date_input("Installment Start Date")

    st.subheader("Credit History")
    default = st.selectbox("Any Previous Default?", ["0", "1"])
    number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, step=1)
    number_of_closed_accounts = st.number_input("Number of Closed Accounts", min_value=0, step=1)
    total_loan_months = st.number_input("Total Loan Months", min_value=0, step=1)
    delinquent_months = st.number_input("Delinquent Months", min_value=0, step=1)
    total_dpd = st.number_input("Total DPD (Days Past Due)", min_value=0, step=1)
    enquiry_count = st.number_input("Credit Enquiry Count", min_value=0, step=1)
    credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", 0, 100, 30)
    # CUR_Class = st.selectbox("CUR Class", ["Poor", "Average", "Good"])

    # Submit button inside the form
    submitted = st.form_submit_button("🔍 Predict Loan Risk")

# -------------------- BUILD INPUT DATA --------------------
if submitted:
    input_data = pd.DataFrame({
        # "cust_id": [cust_id],
        "age": [age],
        "gender": [gender],
        "marital_status": [marital_status],
        "employment_status": [employment_status],
        "income": [income],
        "number_of_dependants": [number_of_dependants],
        "residence_type": [residence_type],
        "years_at_current_address": [years_at_current_address],
        "city": [city],
        "state": [state],
        "zipcode": [zipcode],
        # "loan_id": [loan_id],
        "loan_purpose": [loan_purpose],
        "loan_type": [loan_type],
        "sanction_amount": [sanction_amount],
        "loan_amount": [loan_amount],
        "processing_fee": [processing_fee],
        "gst": [gst],
        "net_disbursement": [net_disbursement],
        "loan_tenure_months": [loan_tenure_months],
        "principal_outstanding": [principal_outstanding],
        "bank_balance_at_application": [bank_balance_at_application],
        # "disbursal_date": [disbursal_date],
        # "installment_start_dt": [installment_start_dt],
        "default": [default],
        "number_of_open_accounts": [number_of_open_accounts],
        "number_of_closed_accounts": [number_of_closed_accounts],
        "total_loan_months": [total_loan_months],
        "delinquent_months": [delinquent_months],
        "total_dpd": [total_dpd],
        "enquiry_count": [enquiry_count],
        "credit_utilization_ratio": [credit_utilization_ratio],
        # "CUR_Class": [CUR_Class]
    })

    for col, encoder in categorical_encoders.items():
        if col in input_data.columns:
            try:
                input_data[col] = encoder.transform(input_data[[col]].astype(str))
            except Exception as e:
                st.error(f"Encoding error in column '{col}': {e}")
    # -------------------- PREDICTION --------------------
    prediction_encoded = model.predict(input_data)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    st.subheader(f"✅ Predicted Loan Application Risk: **{prediction}**")
