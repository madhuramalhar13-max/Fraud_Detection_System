from pathlib import Path
import pandas as pd

DATA_PATH = Path("Data/processed/cleaned_fraud_dataset.csv")
TARGET = "Fraud_Label"


def main():
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("\nOverall fraud rate:",
          round(df[TARGET].mean() * 100, 2), "%")

    categorical = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Card_Type",
        "Previous_Fraudulent_Activity",
    ]

    numeric = [
        "Transaction_Amount",
        "Account_Balance",
        "Daily_Transaction_Count",
        "Card_Age",
    ]

    for col in categorical:
        if col not in df.columns:
            continue

        print(f"\n{'=' * 55}")
        print(f"Fraud rate by {col}")
        result = df.groupby(col, dropna=False)[TARGET].agg(
            Transactions="count",
            Fraud_Rate="mean"
        )
        result["Fraud_Rate"] = (result["Fraud_Rate"] * 100).round(2)
        print(result.sort_values("Fraud_Rate", ascending=False).to_string())

    for col in numeric:
        if col not in df.columns:
            continue

        print(f"\n{'=' * 55}")
        print(f"Fraud rate by {col} range")

        bins = pd.qcut(df[col], q=5, duplicates="drop")
        result = df.groupby(bins, observed=True)[TARGET].agg(
            Transactions="count",
            Fraud_Rate="mean"
        )
        result["Fraud_Rate"] = (result["Fraud_Rate"] * 100).round(2)
        print(result.to_string())

    print("\nInvestigation complete.")


if __name__ == "__main__":
    main()