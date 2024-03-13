# Load the libraries
import streamlit as st
import joblib
import pandas as pd


# Function to load the model
def load_model(file_path):
    '''Load the trained model'''
    model = joblib.load(file_path)
    return model

# Function to give prediction
def predict(model, data):
    '''Make the prediction using the model'''
    prediction = model.predict(data)
    return prediction

# function for the main--- dashboard
def main():
    st.title("Diabetes readmitted prediction")

    # File upload
    st.sidebar.title("Upload data")
    uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('**Sample Data:**')
        st.write(data.head())

        # Load the model
        model = load_model('model/diabetes_model.pkl')

        # make predictions
        prediction = predict(model, data)

        st.write('**Prediction:**')
        st.write(prediction)

if __name__ == '__main__':
    main()