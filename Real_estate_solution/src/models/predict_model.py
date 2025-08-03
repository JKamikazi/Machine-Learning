# evaluate your model
from sklearn.metrics import mean_absolute_error


# Function to predict and evaluate
def eval_model(lrmodel, x_train, x_test, y_train, y_test):
    # make preditions on train set
    train_pred = lrmodel.predict(x_train)
    
    # make predictions om test set
    y_pred = lrmodel.predict(x_test)
    
    # we need mean absolute error
    train_mae = mean_absolute_error(train_pred, y_train)

    #evaluate the model
    test_mae = mean_absolute_error(y_pred, y_test)
    
    return train_mae, test_mae