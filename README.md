# Business Intelligence & Machine Learning App Suite

This repository contains multiple interactive applications built with **Streamlit** to demonstrate machine learning models applied in various business and analytics scenarios.

Each app allows users to interact with trained models through a simple web interface and view predictions and visualizations based on real-world data.

---

## 🔍 Applications Included

### 1. **Loan Eligibility Predictor**
Predicts whether a user is eligible for a loan using a Random Forest classifier trained on financial and demographic data.

**Features:**
- Input form for applicant details
- Real-time eligibility prediction
- Confidence score and feature importance chart

**Model:** Random Forest  
**Dataset:** Cleaned loan application dataset  
**Launch:** `streamlit run loan_app.py`

---

### 2. **University Admission Predictor**
Estimates the probability of admission to a university using academic inputs like GRE, TOEFL, SOP, and CGPA.

**Features:**
- Input GRE, TOEFL, SOP, LOR, CGPA, research, and university rating
- Predicts admission chance using an MLP model
- Probability score shown

**Model:** MLPRegressor  
**Dataset:** UCLA Graduate Admissions  
**Launch:** `streamlit run admission_app.py`

---

### 3. **Customer Segmentation (KMeans Clustering)**
Segments customers into clusters based on age, annual income, and spending score.

**Features:**
- Predicts customer cluster
- Displays cluster scatter plot with your input point
- Silhouette plot for evaluating cluster quality

**Model:** KMeans  
**Dataset:** Mall Customer Dataset  
**Launch:** `streamlit run clustering_app.py`

---

## 🛠️ Technologies Used

- **Streamlit** for app UI
- **Scikit-learn** for modeling
- **Pandas**, **NumPy** for data processing
- **Matplotlib**, **Seaborn** for visualization
- **Pickle** for model storage

---
