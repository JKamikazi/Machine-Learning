import streamlit as st
import pandas as pd
import pickle

# Page title and description
st.title("Real Estate Price Predictor")
st.write("This app predicts the price of a house using a trained machine learning model.")

# Load model
with open("models/LRmodel.pkl", "rb") as file:
    model = pickle.load(file)

# User form
with st.form("input_form"):
    st.subheader("Enter Property Information")

    year_sold = st.number_input("Year Sold", min_value=2000, max_value=2025, value=2023)
    property_tax = st.number_input("Annual Property Tax ($)", min_value=0, step=100)
    insurance = st.number_input("Annual Insurance Cost ($)", min_value=0, step=100)
    beds = st.number_input("Number of Bedrooms", min_value=0, step=1)
    baths = st.number_input("Number of Bathrooms", min_value=0, step=1)
    sqft = st.number_input("Square Footage", min_value=0, step=100)
    year_built = st.number_input("Year Built", min_value=1800, max_value=year_sold, value=2000)
    lot_size = st.number_input("Lot Size (sqft)", min_value=0, step=100)

    basement = st.selectbox("Does the property have a basement?", ["Yes", "No"])
    popular = st.selectbox("Is the property in a popular location?", ["Yes", "No"])
    recession = st.selectbox("Was the property sold during a recession?", ["Yes", "No"])
    property_type = st.selectbox("Property Type", ["Bunglow", "Condo", "Other"])

    submit = st.form_submit_button("Predict Price")

# Prediction
if submit:
    # Binary encoding
    basement_val = 1 if basement == "Yes" else 0
    popular_val = 1 if popular == "Yes" else 0
    recession_val = 1 if recession == "Yes" else 0

    # Derived feature
    property_age = year_sold - year_built

    # One-hot encoding (must match training)
    property_type_Bunglow = 1 if property_type == "Bunglow" else 0
    property_type_Condo = 1 if property_type == "Condo" else 0

    # Final feature vector (exact order)
    input_data = [[
        year_sold, property_tax, insurance, beds, baths, sqft,
        year_built, lot_size, basement_val, popular_val, recession_val,
        property_age, property_type_Bunglow, property_type_Condo
    ]]

    # Predict
    predicted_price = model.predict(input_data)[0]

    # Result
    st.subheader("Predicted House Price")
    st.write(f"${predicted_price:,.2f}")
