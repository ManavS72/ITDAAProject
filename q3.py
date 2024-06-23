# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:06:28 2024

@author: Manav
"""
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset with semicolon delimiter
data = pd.read_csv('heart.csv', delimiter=';')

# Check the columns of the DataFrame
print("Columns in the DataFrame:", data.columns)

# Inspect the first few rows of the DataFrame
print(data.head())

# Ensure the 'target' column exists
if 'target' not in data.columns:
    raise KeyError("The 'target' column is not found in the DataFrame")

# Split data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Standardize numerical features
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'heartDiseaseModel.pkl')
