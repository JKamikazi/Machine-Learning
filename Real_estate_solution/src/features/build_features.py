import pandas as pd

# Performing feature engineering
def feature_eng(df):
    # Create indicator variable for properties with 2 beds and 2 baths
    df['popular']= ((df.beds == 2)&(df.baths == 2)).astype(int)
    
    # Create a new variable recession
    df['recession'] = ((df.year_sold >= 2010) & (df.year_sold<=2013)).astype(int)
    
    # Create a property age feature
    df['property_age'] = df.year_sold - df.year_built
    
    # Remove rows where property_age is less than 0
    df = df[df.property_age >= 0]
    
    # Create dummy variables for 'property_type'
    df = pd.get_dummies(df, columns=['property_type']).astype(int)
    
    # property_type_Condo has dtype uint8, change it to int64
    df.property_type_Condo=(df.property_type_Condo).astype(int)
    
    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_Real_Estate_Dataset.csv', index=None)
    
    # seperate input features in x
    x = df.drop('price', axis=1)

    # store the target variable in y
    y = df['price']
    
    return x, y