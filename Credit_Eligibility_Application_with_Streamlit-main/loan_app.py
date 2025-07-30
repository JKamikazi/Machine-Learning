import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns  # FIX: Seaborn now imported

# Load model
with open("Credit_Eligibility_Application_with_Streamlit-main/models/RFmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")


# Page Title
st.title("Credit Loan Eligibility Predictor")

# Input Form
st.subheader("Enter Applicant Details")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Marital Status", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=1000)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=1000)
        LoanAmount = st.number_input("Loan Amount", min_value=0, step=1000)
        Loan_Amount_Term = st.selectbox("Loan Term (Months)", ["360", "180", "240", "120", "60"])
        Credit_History = st.selectbox("Credit History", ["1", "0"])

    submit = st.form_submit_button("Predict Loan Eligibility")

# Encoding function
def prepare_input():
    Gender_Male = 0 if Gender == "Female" else 1
    Gender_Female = 1 if Gender == "Female" else 0

    Married_Yes = 1 if Married == "Yes" else 0
    Married_No = 1 if Married == "No" else 0

    Dependents_0 = 1 if Dependents == "0" else 0
    Dependents_1 = 1 if Dependents == "1" else 0
    Dependents_2 = 1 if Dependents == "2" else 0
    Dependents_3 = 1 if Dependents == "3+" else 0

    Education_Graduate = 1 if Education == "Graduate" else 0
    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
    Self_Employed_No = 1 if Self_Employed == "No" else 0

    Property_Area_Rural = 1 if Property_Area == "Rural" else 0
    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0

    return [[
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        int(Loan_Amount_Term), int(Credit_History), Gender_Female, Gender_Male,
        Married_No, Married_Yes, Dependents_0, Dependents_1,
        Dependents_2, Dependents_3, Education_Graduate,
        Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
        Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
    ]]

# Prediction and Output
if submit:
    X = prepare_input()
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")

    if prediction == "Y":  # FIXED: prediction was being compared to 'Y'
        st.success(" You are eligible for the loan!")
    else:
        st.error("Sorry, you are not eligible for the loan.")

    st.progress(int(prob * 100))
    st.write(f"Model Confidence: **{round(prob * 100, 2)}%**")

    # Feature importance
    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        features = [
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Term", "Credit_History",
            "Gender_Female", "Gender_Male", "Married_No", "Married_Yes", "Dep_0", "Dep_1",
            "Dep_2", "Dep_3+", "Edu_Grad", "Edu_NotGrad", "SelfEmp_No", "SelfEmp_Yes",
            "Area_Rural", "Area_Semiurban", "Area_Urban"
        ]
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="Blues_d", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display feature importances: {e}")
