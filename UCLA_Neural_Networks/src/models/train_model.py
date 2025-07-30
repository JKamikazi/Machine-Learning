from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle


# Function to train the model
def train_MLPmodel(x, y):
    # Splitting the dataset into train and test data
    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123)

    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    
    # Now transform xtrain and xtest
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    
    # del MLP
    
    # fit/train the model. Check batch size.
    MLP = MLPClassifier(hidden_layer_sizes=(2,2), batch_size=50, max_iter=200)
    MLP.fit(xtrain_scaled,ytrain)


    # Save the trained model
    with open('models/MLPmodel.pkl', 'wb') as f:
        pickle.dump(MLP, f)

    return MLP, xtest_scaled, ytest
