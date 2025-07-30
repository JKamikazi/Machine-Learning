import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_feature_coefficients(model, feature_names):
    """
    Plot the coefficients of a linear regression model.

    Args:
        model: Trained LinearRegression model.
        feature_names (list): List of feature names.
    """
    coefs = model.coef_
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    coef_df = coef_df.sort_values(by='Coefficient')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted values for regression.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.show()
