import streamlit as st
import numpy as np
import pickle

# Title and intro
st.title("University Admission Predictor")
st.write("Enter your academic profile to estimate your admission chances.")

# Load model
with open("models/MLPmodel.pkl", "rb") as file:
    model = pickle.load(file)

# Input form
with st.form("admission_form"):
    st.subheader("Your Profile")

    GRE_Score = st.number_input("GRE Score (out of 340)", min_value=260, max_value=340, value=300, step=1)
    TOEFL_Score = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=100, step=1)
    SOP = st.slider("SOP Rating", 1.0, 5.0, step=0.5)
    LOR = st.slider("LOR Rating", 1.0, 5.0, step=0.5)
    CGPA = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.01)
    
    University_Rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    Research = st.selectbox("Research Experience", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# Handle input and prediction
if submitted:
    # One-hot encoding for University_Rating
    uni_encoded = [1 if University_Rating == i else 0 for i in range(1, 6)]

    # One-hot for Research
    research_0 = 1 if Research == "No" else 0
    research_1 = 1 if Research == "Yes" else 0

    # Feature vector: matches model input
    input_vector = np.array([[
        GRE_Score,
        TOEFL_Score,
        SOP,
        LOR,
        CGPA,
        *uni_encoded,
        research_0,
        research_1
    ]])

    # Predict
    prediction = model.predict(input_vector)[0]
    percentage = round(prediction * 100, 2)

    # Output
    st.subheader("Your Predicted Admission Chance:")
    st.write(f"{percentage}%")
