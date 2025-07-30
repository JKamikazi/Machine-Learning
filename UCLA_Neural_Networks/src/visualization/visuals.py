
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Admit Chance")
    plt.ylabel("Predicted Admit Chance")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

def plot_feature_effects(model, feature_names):
    if hasattr(model, "coefs_"):
        weights = model.coefs_[0].mean(axis=1)
        df = pd.DataFrame({"Feature": feature_names, "Weight": weights})
        df = df.sort_values("Weight")

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Weight", y="Feature", data=df)
        plt.title("Average Input Layer Weights (MLP)")
        plt.tight_layout()
        plt.show()
