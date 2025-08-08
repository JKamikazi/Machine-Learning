from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd 
import plotly.express as px

#set Steamlit layout to wide
st.set_page_config(layout="wide")

#load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

#load the MinMaxScaler
with open('scaler.pkl', 'rb')as file:
    scaler = pickle.load(file)

#define the input features for the model
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

#columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

#updated default values
default_values = [
    600, 30, 2, 8000, 2, 6000,
    True, False, False, True, False, False, True, False, True
]

#Sidebar setup
st.sidebar.header("User Inputs")

#collect user inputs
user_inputs  =  {}
for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1 if isinstance(default_values[i], int) else 0.01
        )
    elif isinstance(default_values[i], bool):
        user_inputs[feature] = st.sidebar.checkbox(feature,value=default_values[i])
    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1
        )
        
#covert inputs to a dataframe
input_data = pd.DataFrame([user_inputs])

#Apply MinMaxScaler to the required columns
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

#App header title
st.title("Customer Churn Prediction")
#Page layout 
left_col, right_col = st.columns(2)

#left page: Feature importance
with left_col:
    st.header("Feature Importance")
    #load feature importance data from the excel file
    feature_importance_df = pd.read_excel("feature_importance.xlsx", usecols=["Feature", "Feature Importance Score"])
    #plot the feature importance bar chart
    fig = px.bar(
        feature_importance_df.sort_values(by="Feature Importance Score", ascending=True),
        x="Feature Importance Score",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        labels={"Feature Importance": "Importance", "Feature": "Features"},
        width=400,  #set custom width
        height=500,  #set custom height
    )
    st.plotly_chart(fig)

#Right page: prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        #get the prediction probabilities and label
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        #map prediction to label
        prediction_label = "Churned" if prediction == 1 else "Retain"

        #display results
        st.subheader(f"Predicted Value: {prediction_label}")
        st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
        st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
        #display a clear output for the prediction
        st.markdown(f"### Output: **{prediction_label}**")

    
#Streamlit run churn_prediction_app.py   