from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import pickle


# Function to train the model
def train_predict_Kmodel(df):

    # try using a for loop for WCSS
    k = range(3,9)
    K = []
    WCSS = []
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
        
    # Store the number of clusters and their respective WSS scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    
    
    # Train a model on 'Age','Annual_Income','Spending_Score' features
    k = range(3,9)
    K = []
    ss = []
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Age','Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
        
    # Store the number of clusters and their respective silhouette scores in a dataframe
    wss['Silhouette_Score']=ss

    # Save the trained model
    with open('models/Kmodel.pkl', 'wb') as f:
        pickle.dump(kmodel, f)

    return wss
