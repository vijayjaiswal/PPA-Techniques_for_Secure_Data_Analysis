# ------------------------------------------------------------
# Basic PII Scanning in Fake People Dataset
# ------------------------------------------------------------
import pandas as pd

# Downloaded and renamed from https://www.kaggle.com/datasets/shuvokumarbasak4004/fake-dataset-for-practice?select=datainfo.csv

# Load a sample dataset (e.g., fake_people_datainfo file)
df = pd.read_csv('.\\data\\fake_people_datainfo.csv')

# Simple check for potential PII columns based on column names
pii_keywords = ['name', 'email', 'ssn', 'address', 'phone', 'dob']
potential_pii_columns = []

for column in df.columns:
    for keyword in pii_keywords:
        # Match for exact name
        if keyword in column.lower():
            potential_pii_columns.append(column)
            break
# Print the result
print("Potential PII columns based on names:", potential_pii_columns)