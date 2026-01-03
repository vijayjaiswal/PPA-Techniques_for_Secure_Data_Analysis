import pandas as pd

# Sample: Load the UCI Adult dataset (update path or URL as needed)
columns = [
    "record_id", "full_name", "email", "phone_number", "national_id", "age", "gender", 
    "zip_code", "city", "occupation", "medical_condition", "annual_income"
]
# Load dataset
df = pd.read_csv(".\dataset.csv")

# df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Display basic info
print(f"Dataset loaded with {df.shape[0]} records and {df.shape[1]} columns.\n")

# Define common PII regex patterns
PII_PATTERNS = {
    "email": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
    "phone_number": r"^\+?\d{10,15}$",
    "national_id": r"^\d{12}$"
}

results = []

for col in df.columns:
    series = df[col].dropna().astype(str)

    # Uniqueness ratio
    uniqueness_ratio = series.nunique() / max(len(series), 1)

    # Pattern matching
    matched_patterns = [
        name for name, pattern in PII_PATTERNS.items()
        if series.str.match(pattern).any()
    ]

    results.append({
        "column": col,
        "uniqueness_ratio": round(uniqueness_ratio, 2),
        "pii_patterns_detected": matched_patterns,
        "possible_identifier": uniqueness_ratio > 0.9 or bool(matched_patterns)
    })

# Summary dataframe
identifier_report = pd.DataFrame(results)

print(identifier_report.sort_values(
    by=["possible_identifier", "uniqueness_ratio"],
    ascending=False
))
