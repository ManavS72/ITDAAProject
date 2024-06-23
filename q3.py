# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:06:28 2024

@author: Manav
"""
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

conn = sqlite3.connect('heartDisease.db')
heartData = pd.read_sql_query("SELECT * FROM heartDisease", conn)
conn.close()


X = heartData.drop(columns='target')
y = heartData['target']

# Standardize numerical features
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Initialize models
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Train models
log_reg.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf_clf = rf_clf.predict(X_test)
y_pred_svm_clf = svm_clf.predict(X_test)

# Evaluate models
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(classification_report(y_test, y_pred_log_reg))

print("\nRandom Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_clf)}")
print(classification_report(y_test, y_pred_rf_clf))

print("\nSupport Vector Machine:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm_clf)}")
print(classification_report(y_test, y_pred_svm_clf))

# Save the best model to disk
best_model = rf_clf  
joblib.dump(best_model, 'heartDiseaseModel.pkl')

