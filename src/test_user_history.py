from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("Data/processed/cleaned_fraud_dataset.csv")


def main():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    print("Fraud rate by previous fraud activity:")
    print(
        (df.groupby("Previous_Fraudulent_Activity")["Fraud_Label"]
        .agg(["count", "mean"]).assign(
            fraud_rate_percent=lambda x: (x["mean"] * 100).round(2)
        )).to_string()
    )

    # Test the existing features without transaction/user IDs.
    X = df.drop(
        columns=["Fraud_Label", "Transaction_ID", "User_ID", "Date"],
        errors="ignore",
    )
    y = df["Fraud_Label"].astype(int)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        ), numeric),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ), categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nExisting-feature baseline:")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, probabilities):.4f}")

    print("\nNote: This is a random stratified split, not a time-based test.")


if __name__ == "__main__":
    main()