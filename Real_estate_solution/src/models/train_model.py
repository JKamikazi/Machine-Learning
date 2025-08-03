# import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


# Function to train the model
def train_lrmodel(x, y):
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234, stratify=x.property_type_Condo)
    
    # train your model
    lrmodel = LinearRegression().fit(x_train,y_train)
    
    # Save the trained model
    with open('models/LRmodel.pkl', 'wb') as f:
        pickle.dump(lrmodel, f)
        
    return lrmodel, x_train, x_test, y_train, y_test 