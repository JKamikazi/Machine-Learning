
import pandas as pd

def load_data(data_path):
    
    # Import the data from 'credit.csv'
    df = pd.read_csv(data_path)

    return df
