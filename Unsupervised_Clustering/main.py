# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Swapnil Kangralkar',
#     license='',
# )

from src.data.make_dataset import load_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance
# from src.features.build_features import create_dummy_vars
from src.models.train_and_predict_model import train_predict_Kmodel
# from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/mall_customers.csv"
    df = load_data(data_path)

    # Create dummy variables and separate features and target
    # X, y = create_dummy_vars(df)

    # Train the logistic regression model
    kmodel = train_predict_Kmodel(df)

    # Evaluate the model
    plot_feature_importance(kmodel)
    # accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
    print(kmodel)
