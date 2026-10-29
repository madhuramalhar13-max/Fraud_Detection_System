from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model_utils import prepare_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "processed" / "cleaned_fraud_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Fraud_Label"].value_counts(normalize=True))

# Check whether numeric features differ between classes.
print("\nNUMERIC FEATURE COMPARISON")
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    if column == "Fraud_Label":
        continue

    legitimate_mean = df.loc[
        df["Fraud_Label"] == 0, column
    ].mean()

    fraud_mean = df.loc[
        df["Fraud_Label"] == 1, column
    ].mean()

    print(
        f"{column}: legitimate mean={legitimate_mean:.3f}, "
        f"fraud mean={fraud_mean:.3f}"
    )

# Check whether labels are associated with categorical features.
print("\nCATEGORICAL FEATURE FRAUD RATES")

for column in df.select_dtypes(exclude="number").columns:
    if column in ["Transaction_ID", "User_ID", "Date"]:
        continue

    rates = df.groupby(column, dropna=False)["Fraud_Label"].mean()

    print(f"\n{column}")
    print(rates.sort_values(ascending=False).head(10))

# Evaluate whether the features carry useful ranking information.
X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print("\nFEATURE SIGNAL CHECK")
print("Fraud prevalence:", round(y_test.mean(), 4))
print("Average precision (PR-AUC):",
      round(average_precision_score(y_test, probabilities), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("\nTARGET ASSOCIATION CHECK")

for column in df.select_dtypes(include="number").columns:
    if column in ["Fraud_Label", "Transaction_ID", "User_ID"]:
        continue

    correlation = df[column].corr(df["Fraud_Label"])

    print(f"{column}: correlation = {correlation:.4f}")

print("\nTARGET PATTERN CHECK")

print(
    df.groupby("Fraud_Label")[
        ["Transaction_Amount", "Account_Balance",
         "Daily_Transaction_Count", "Card_Age"]
    ].describe().round(2)
)