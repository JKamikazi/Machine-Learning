
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

def plot_clusters_2d(data, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    df = pd.DataFrame(components, columns=["PC1", "PC2"])
    df["Cluster"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
    plt.title("2D Cluster Plot (PCA Reduced)")
    plt.tight_layout()
    plt.show()

def plot_elbow(distortions, k_range):
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, distortions, marker='o')
    plt.xticks(k_range)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Distortion")
    plt.title("Elbow Method for Optimal K")
    plt.tight_layout()
    plt.show()
