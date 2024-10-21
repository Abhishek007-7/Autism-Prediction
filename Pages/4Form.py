import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the autism dataset
st.title(":bookmark_tabs: :blue[Autism Data Assessment]")
st.write("---")
st.write("Fill the form below to check if your child is suffering from ASD ")

# Try to load the dataset and handle errors
try:
    autism_dataset = pd.read_csv('train.csv')
except FileNotFoundError:
    st.error("The dataset 'train.csv' could not be found. Please ensure it is in the same directory as this script.")
    st.stop()  # Stop execution if the dataset is not found

# Separating the data and labels
X = autism_dataset.drop(columns='Class/ASD', axis=1)
Y = autism_dataset['Class/ASD']

# One-hot encode categorical variables in X
X = pd.get_dummies(X, drop_first=True)

# Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Function to convert string inputs to numerical values
def ValueCount(value):
    return 1 if value == "Yes" else 0

def Sex(value):
    return 1 if value == "Female" else 0

# Function to preprocess input data
def preprocess_input_data(input_data, feature_columns):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    # One-hot encode the input data
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    # Align the columns with the training data
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Check if input_encoded has the same number of features as the scaler expects
    if input_encoded.shape[1] != len(feature_columns):
        st.error("Mismatch in number of features. Please check your inputs.")
        return None  # Return None or handle as needed

    return input_encoded

# Form layout
d1 = list(range(11))  # 0 to 10
val1 = st.selectbox("Social Responsiveness", d1)

d2 = list(range(19))  # 0 to 18
val2 = st.selectbox("Age", d2)

d3 = ["No", "Yes"]
val3 = ValueCount(st.selectbox("Speech Delay", d3))
val4 = ValueCount(st.selectbox("Learning disorder", d3))
val5 = ValueCount(st.selectbox("Genetic disorders", d3))
val6 = ValueCount(st.selectbox("Depression", d3))
val7 = ValueCount(st.selectbox("Intellectual disability", d3))
val8 = ValueCount(st.selectbox("Social/Behavioural issues", d3))
val9 = ValueCount(st.selectbox("Childhood Autism Rating Scale", d1))
val10 = ValueCount(st.selectbox("Anxiety disorder", d3))

d4 = ["Female", "Male"]
val11 = Sex(st.selectbox("Gender", d4))

val12 = ValueCount(st.selectbox("Suffers from Jaundice", d3))
val13 = ValueCount(st.selectbox("Family member history with ASD", d3))

# Input data
input_data = [val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13]

# Get the feature columns from the training set after one-hot encoding
feature_columns = X.columns.tolist()  # This line has been moved below data preparation

# Preprocess the input data
input_encoded = preprocess_input_data(input_data, feature_columns)

# Only standardize if input_encoded is not None
if input_encoded is not None:
    # Standardize the input data
    std_data = scaler.transform(input_encoded)

    # Prediction
    prediction = classifier.predict(std_data)

    # Display results
    with st.expander("Analyze Provided Data"):
        st.subheader("Results:")
        if prediction[0] == 0:
            st.info('The person is not with Autism Spectrum Disorder.')
        else:
            st.warning('The person is with Autism Spectrum Disorder.')
