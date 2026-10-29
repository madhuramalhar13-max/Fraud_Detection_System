from pathlib import Path
import pandas as pd

PATH = Path("data/synthetic_fraud_dataset1.csv")

def main():
    df = pd.read_csv(PATH)

    print("Columns:", df.columns.tolist())
    print("\nFraud label distribution:")
    print(df["Fraud_Label"].value_counts().sort_index())

    print("\nPrevious fraud activity vs fraud label:")
    print(pd.crosstab(
        df["Previous_Fraudulent_Activity"],
        df["Fraud_Label"],
        margins=True
    ))

    print("\nFraud rate by previous fraud activity:")
    print(
        df.groupby("Previous_Fraudulent_Activity")["Fraud_Label"]
        .agg(["count", "mean"])
        .assign(fraud_rate_percent=lambda x: (x["mean"] * 100).round(2))
        .to_string()
    )

    print("\nSample fraudulent transactions:")
    print(
        df.loc[df["Fraud_Label"] == 1]
        .head(10)
        .to_string(index=False)
    )

    print("\nSample legitimate transactions:")
    print(
        df.loc[df["Fraud_Label"] == 0]
        .head(10)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()