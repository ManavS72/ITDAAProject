# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:53:08 2024

@author: Manav
"""

import sqlite3
import pandas as pd

file_path = 'heart.csv'
heartData = pd.read_csv(file_path, delimiter=';')


conn = sqlite3.connect('heartDisease.db')


heartData.to_sql('heartDisease', conn, if_exists='replace', index=False)


cursor = conn.cursor()
cursor.execute("SELECT * FROM heartDisease LIMIT 10")
rows = cursor.fetchall()
for row in rows:
    print(row)


conn.close()
