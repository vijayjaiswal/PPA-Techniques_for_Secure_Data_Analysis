import pandas as pd
import numpy as np

# Load a sample dataset (e.g., an employee file)
# Downloaded and renamed from https://www.kaggle.com/datasets/shuvokumarbasak4004/fake-dataset-for-practice?select=datainfo.csv

df = pd.read_csv('D:\\Vijay_Personal\\workspace\\Projects\\Author_Books\\Code\\PPA-Techniques_for_Secure_Data_Analysis\\Chapter_01\data\\fake_people_datainfo.csv')

# Simple check for potential PII columns based on column names
pii_keywords = ['name', 'email', 'ssn', 'address', 'phone', 'dob']
potential_pii_columns = []

for column in df.columns:
    for keyword in pii_keywords:
        if keyword in column.lower():
            potential_pii_columns.append(column)
            break

print("Potential PII columns based on names:", potential_pii_columns)