import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Mall Customer Segmentation Model")
st.write("Predict customer group using Age, Income, and Spending Score.")

# Load model
with open("models/Kmodel.pkl", "rb") as file:
    model = pickle.load(file)

# Input form
with st.form("input_form"):
    st.subheader("Enter Customer Details")

    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Annual Income (k$)", 15, 150, 60)
    score = st.slider("Spending Score (1–100)", 1, 100, 50)

    submitted = st.form_submit_button("Predict Cluster")

# Prediction logic
if submitted:
    input_data = np.array([[age, income, score]])
    cluster = model.predict(input_data)[0]

    st.subheader("Predicted Cluster")
    st.write(f"This customer belongs to **Cluster {cluster}**")

    # Try loading full dataset for plotting
    try:
        df = pd.read_csv("data/mall_customers.csv")
        df = df.drop(columns=['Customer_ID', 'Gender'], errors='ignore')
        features = df[['Age', 'Annual_Income', 'Spending_Score']]
        df['Cluster'] = model.predict(features)

        # Plot the clusters
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x='Annual_Income',
            y='Spending_Score',
            hue='Cluster',
            palette='colorblind',
            data=df,
            ax=ax
        )

        # Plot the new customer input
        ax.scatter(income, score, color='black', s=100, label='You', marker='X')
        ax.legend()
        ax.set_title("Customer Clusters (Income vs Spending Score)")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not generate cluster plot: {e}")
        
#streamlit run clustering_app.py