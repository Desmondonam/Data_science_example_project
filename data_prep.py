# Import the libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a function to load the data
def load_data(file_path):
    '''Load the diabetic dataset.'''
    data = pd.read_csv(file_path)
    return data

# Create a function to preprocess the data
def preprocess_data(data):
    '''preprocess teh data that has been loaded'''
    #Drop the irrelevant columns in the data, or handle the missing values in the data
    # Let us assume our data is clean

    # Split the data into features and target
    X = data.drop('readmitted', axis = 1)
    y = data['readmitted']

    # Split the data inot the training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test