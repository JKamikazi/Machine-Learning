# unsupervised_clustering_app

This app is built with Streamlit and demonstrates an unsupervised learning model (KMeans) applied to a dataset for customer segmentation or feature grouping.

[Launch the App](https://clustering-a.streamlit.app/)

## Features
- Upload or view data
- Automatically clusters input data using KMeans
- Interactive scatter plot showing clustered data points
- Visualization of cluster centroids and labels

## Dataset
The dataset used includes various numeric attributes used to group observations based on feature similarity. Common use cases include:
- Customer segmentation
- Feature grouping
- Anomaly detection

## Technologies
- **Streamlit** for the frontend app
- **Scikit-learn** for KMeans clustering
- **Pandas**, **NumPy** for data handling
- **Matplotlib**, **Seaborn**, **Plotly** for visualization

## How to Run
```bash
git clone https://github.com/your-username/unsupervised_clustering_app.git
cd unsupervised_clustering_app

python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

pip install -r requirements.txt
streamlit run clustering_app.py
```

## Future Additions
- Cluster validation metrics (e.g., silhouette score)
- Auto-tuning number of clusters
- Export clustered results
