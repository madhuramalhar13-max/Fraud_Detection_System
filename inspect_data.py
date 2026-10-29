import pandas as pd
from pathlib import Path

data_path = Path("Data/processed/cleaned_fraud_dataset.csv")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

if "Fraud_Label" in df.columns:
    print("\nFraud distribution:")
    print(df["Fraud_Label"].value_counts(dropna=False))
else:
    print("\nFraud_Label column not found!")

    print("\nFraud percentage:")
print(
    df["Fraud_Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)