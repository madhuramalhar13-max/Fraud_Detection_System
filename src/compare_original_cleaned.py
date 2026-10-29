from pathlib import Path
import pandas as pd

original_path = Path("data/synthetic_fraud_dataset1.csv")
cleaned_path = Path("Data/processed/cleaned_fraud_dataset.csv")

original = pd.read_csv(original_path)
cleaned = pd.read_csv(cleaned_path)

print("Original shape:", original.shape)
print("Cleaned shape:", cleaned.shape)

if list(original.columns) != list(cleaned.columns):
    print("\nColumn names or order differ.")
else:
    print("\nColumn names and order match.")

if original.equals(cleaned):
    print("RESULT: Original and cleaned datasets are identical.")
else:
    print("RESULT: Differences found between the datasets.")

    for column in original.columns:
        if not original[column].equals(cleaned[column]):
            print(f"\nDifferences in column: {column}")
            print("Original sample:", original[column].head(3).tolist())
            print("Cleaned sample:", cleaned[column].head(3).tolist())

if original.shape == cleaned.shape:
    print("\nFraud labels identical:",
          original["Fraud_Label"].equals(cleaned["Fraud_Label"]))

print("\nComparison complete.")