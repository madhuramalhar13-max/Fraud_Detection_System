from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = Path("Data/processed/cleaned_fraud_dataset.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Create the features recommended in the handover report.
    dates = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount_to_Balance_Ratio"] = (
        df["Transaction_Amount"] /
        (df["Account_Balance"] + 1.0)
    )
    df["Month"] = dates.dt.month
    df["Day_Of_Week"] = dates.dt.dayofweek
    df["Is_Weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    y = df["Fraud_Label"]
    X = df.drop(
        columns=["Fraud_Label", "Transaction_ID", "User_ID", "Date"]
    )

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessing = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nENGINEERED FEATURE TEST")
    print("Features:", X.columns.tolist())
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("PR-AUC:", round(
        average_precision_score(y_test, probabilities), 4
    ))
    print("PR-AUC baseline:", round(y_test.mean(), 4))


if __name__ == "__main__":
    main()