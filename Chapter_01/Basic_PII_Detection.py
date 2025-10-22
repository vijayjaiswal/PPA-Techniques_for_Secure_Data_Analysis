# ------------------------------------------------------------
# Basic PII Detection in UCI Adult Income Dataset
# ------------------------------------------------------------
import pandas as pd
import re

# Sample: Load the UCI Adult dataset (update path or URL as needed)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Display basic info
print(f"Dataset loaded with {df.shape[0]} records and {df.shape[1]} columns.\n")

# ------------------------------------------------------------
# Define regex patterns for common PII types
# ------------------------------------------------------------
PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b",
    "address": r"\d{1,4}\s+[A-Za-z0-9\s.,'-]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\.?",
    "id_number": r"\b\d{9,12}\b",
    "name": r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b"
}

# ------------------------------------------------------------
# Function to check if a value contains potential PII
# ------------------------------------------------------------
def detect_pii_in_value(value):
    if pd.isna(value):
        return None
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, str(value)):
            return pii_type
    return None

# ------------------------------------------------------------
# Apply detection across all string columns
# ------------------------------------------------------------
pii_summary = {}

for col in df.columns:
    matches = df[col].astype(str).apply(detect_pii_in_value).dropna()
    if not matches.empty:
        pii_summary[col] = matches.value_counts().to_dict()

# ------------------------------------------------------------
# Display summary of detected PII
# ------------------------------------------------------------
if pii_summary:
    print("Possible PII detected in the dataset:")
    for col, details in pii_summary.items():
        print(f" - {col}: {details}")
else:
    print("No potential PII detected in the dataset.")
