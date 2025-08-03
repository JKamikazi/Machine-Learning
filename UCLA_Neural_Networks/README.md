# admission_chance_predictor

This app is built using Streamlit and deployed on Streamlit Community Cloud.

[Visit the app here](https://admission-chance-predictor.streamlit.app/)

**password**: `streamlit`

This application predicts a student's chance of admission to a university based on their academic profile using a trained MLP (Neural Network) model.

## Features
- User-friendly interface to input GRE, TOEFL, CGPA, and more
- Real-time prediction of admission probability
- Hosted online and ready for use

## Dataset
The model uses the UCLA Admission dataset, which includes:
- GRE Score
- TOEFL Score
- University Rating
- SOP Strength
- LOR Strength
- CGPA
- Research Experience

## Technologies Used
- **Streamlit** for the web app
- **Scikit-learn** and **MLPRegressor** for modeling
- **NumPy**, **Pandas** for data handling
- **Matplotlib**, **Seaborn** for visualizations

## Visualizations
- Correlation heatmap
- Feature weights (coefficients)
- Actual vs. Predicted scatter plot

## How to Run Locally
```bash
git clone https://github.com/your-username/admission_chance_predictor.git
cd admission_chance_predictor

python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

pip install -r requirements.txt
streamlit run admission_app.py
```

## Future Plans
- Add SHAP explainability
- Track prediction history
- Export results
