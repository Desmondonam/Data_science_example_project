# Import neccesary libraries
from sklearn.linear_model import LogisticRegression
import joblib

# Create a function to train the model
def train_model(X_train, y_train):
    '''Train the logistic regression model'''
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Save the model
def save_model(model, file_path):
    '''save the model'''
    joblib.dump('diabetes.pkl', 'models')