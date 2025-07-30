# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix

# # Function to predict and evaluate
def evaluate_model(MLP, xtest_scaled, ytest):
    # make Predictions
    ypred = MLP.predict(xtest_scaled)

    # Calculate the accuracy score
    accuracy = accuracy_score(ytest, ypred)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(ytest, ypred)

    return accuracy, confusion_mat