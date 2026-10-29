from pathlib import Path

import pandas as pd

DATA_PATH = Path("Data/processed/cleaned_fraud_dataset.csv")
TARGET = "Fraud_Label"

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET QUALITY AUDIT ==========")
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    if TARGET not in df.columns:
        print(f"\nERROR: Target column '{TARGET}' is missing.")
        return

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df[TARGET].value_counts(dropna=False))
    print("\nTarget percentages:")
    print(
        (df[TARGET].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\n========== DUPLICATE ROWS ==========")
    print("Complete duplicate rows:", df.duplicated().sum())

    feature_cols = [
        col for col in df.columns
        if col not in [TARGET, "Transaction_ID", "User_ID"]
    ]

    print("\n========== NUMERIC FEATURE CORRELATIONS ==========")
    numeric_cols = df[feature_cols].select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_cols:
        print(
            df[numeric_cols + [TARGET]]
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .sort_values(key=abs, ascending=False)
            .round(4)
        )

    print("\n========== FRAUD RATE BY CATEGORY ==========")
    categorical_cols = df[feature_cols].select_dtypes(
        exclude="number"
    ).columns.tolist()

    for col in categorical_cols:
        print(f"\n{col}:")
        summary = (
            df.groupby(col, dropna=False)[TARGET]
            .agg(["count", "mean"])
            .sort_values("mean", ascending=False)
        )
        summary["fraud_rate_percent"] = (
            summary["mean"] * 100
        ).round(2)
        print(summary.drop(columns="mean").to_string())

    print("\n========== AUDIT COMPLETE ==========")

if __name__ == "__main__":
    main()