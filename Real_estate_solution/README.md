# real_estate_price_predictor
This app is built using Streamlit and deployed via Streamlit Community Cloud.

[Visit the app here](https://real-estate-price-predictor.streamlit.app/)

password - streamlit

This application predicts the price of a house based on key property features using a machine learning model trained on a historical real estate dataset.

## Features
- Simple, user-friendly Streamlit interface.
- Input form for property details like size, location, age, and more.
- Real-time prediction of house prices based on trained model.
- Easily accessible via web.

## Dataset
The model is trained on a real estate dataset with the following key features:
- Year Sold
- Property Tax
- Insurance Cost
- Bedrooms
- Bathrooms
- Square Footage
- Year Built
- Lot Size
- Basement (Yes/No)
- Popular Location (Yes/No)
- Recession Period (Yes/No)
- Property Age
- Property Type (Bunglow, Condo)

## Technologies Used
- **Streamlit**: For building the app interface.
- **Scikit-learn**: For training and loading the regression model.
- **Pandas** and **NumPy**: For data handling and preprocessing.
- **Matplotlib** and **Seaborn**: For any optional data visualization.

## Model
The app uses a Linear Regression model trained on cleaned and encoded real estate features. It includes engineered features like property age and one-hot encoded categories for property type.

## Future Enhancements
- Add SHAP explainability for model insights.
- Visualize predictions vs actual data trends.
- Support alternative property datasets.

## Installation (for local deployment)
To run locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real_estate_price_predictor.git
   cd real_estate_price_predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

#### Thank you for using the Real Estate Price Predictor! Share your thoughts and improvements.
