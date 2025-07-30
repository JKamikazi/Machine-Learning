import pandas as pd

def load_data(data_path):
    # Import the data from 'real_estate.csv'
    data = pd.read_csv(data_path)
    
    # Converting the target variable into a categorical variable
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    
    # Dropping columns
    data = data.drop(['Serial_No'], axis=1)
    
    # convert to categorical data
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')
    
    return data