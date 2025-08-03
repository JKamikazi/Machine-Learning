import pandas as pd

def load_data(data_path):
    # Import the data from 'real_estate.csv'
    df = pd.read_csv(data_path)
    
    # Replace missing basement values to 0
    # inplace=True updates the values in the original DataFrame.
    #df['basement']= df['basement'].fillna(0)
    df['basement'].fillna(0, inplace=True)
    
    # Drop the row with lot_size = 1220551
    df = df.drop(102)
    
    return df