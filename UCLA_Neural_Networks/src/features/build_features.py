import pandas as pd

# create dummy features
def create_dummy_vars(data):
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    clean_data = pd.get_dummies(data, columns=['University_Rating','Research'],dtype='int')
    
    # store the processed dataset in data/processed
    clean_data.to_csv('data/processed/Processed_Admission_Dataset.csv', index=None)

    # Separate the input features and target variable
    x = clean_data.drop(['Admit_Chance'], axis=1)
    y = clean_data['Admit_Chance']

    return x, y