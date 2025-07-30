# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Leonard Umoru',
#     license='',
# )

from src.data.make_dataset import load_data
# from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import feature_eng
from src.models.train_model import train_lrmodel
from src.models.predict_model import eval_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/real_estate.csv"
    df = load_data(data_path)

    # Create dummy variables and separate features and target
    x, y = feature_eng(df)

    # Train the linear regression model
    lrmodel, x_train, x_test, y_train, y_test = train_lrmodel(x, y)

    # Evaluate the model
    # plot_feature_importance(lrmodel, x)
    train_mae, test_mae = eval_model(lrmodel, x_train, x_test, y_train, y_test)
    print(f"Train error is, {train_mae}")
    print(f"Test error is, {test_mae}")
