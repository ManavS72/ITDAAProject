# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:03:34 2024

@author: Manav
"""

import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('heartDiseaseModel.pkl')

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

# Define the app title
st.title("Heart Disease Prediction")

# Input fields for patient details
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], format_func=lambda x: ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"][x])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("The Slope of the Peak Exercise ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

# Make prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Standardize numerical features
    numerical_indices = [0, 3, 4, 7, 9]
    input_data[:, numerical_indices] = scaler.transform(input_data[:, numerical_indices])
    
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"The patient is likely to have heart disease. (Probability: {prediction_prob[0][1]:.2f})")
    else:
        st.success(f"The patient is unlikely to have heart disease. (Probability: {prediction_prob[0][0]:.2f})")

# Add error handling
try:
    # Code that may raise exceptions
    pass
except Exception as e:
    st.error(f"An error occurred: {e}")
