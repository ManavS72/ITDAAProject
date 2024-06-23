# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:30:51 2024

@author: Manav
"""

import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

conn = sqlite3.connect('heartDisease.db')


heartData = pd.read_sql_query("SELECT * FROM heartDisease", conn)

conn.close()

# Checking for missing values
print(heartData.isnull().sum())

# Standardize the numerical features
numericalFeatures = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
heartData[numericalFeatures] = scaler.fit_transform(heartData[numericalFeatures])

# Display the cleaned data
heartData.head()


# Plot the distribution of classes for the categorical variables based on the target variable
categoricalFeatures = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

plt.figure(figsize=(20, 15))

for i, feature in enumerate(categoricalFeatures, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=heartData, x=feature, hue='target')
    plt.title(f'Distribution of {feature} by target')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))

for i, feature in enumerate(numericalFeatures, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=heartData, x=feature, kde=True, hue='target', multiple="stack")
    plt.title(f'Distribution of {feature} by target')

plt.tight_layout()
plt.show()

