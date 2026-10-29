from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = Path("Data/processed/cleaned_fraud_dataset.csv")
TARGET = "Fraud_Label"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[
        TARGET, "Transaction_ID", "User_ID"
    ], errors="ignore")
    y = df[TARGET]

    if "Date" in X.columns:
        dates = pd.to_datetime(X["Date"], errors="coerce")
        X["Transaction_Month"] = dates.dt.month
        X["Transaction_Day"] = dates.dt.day
        X["Transaction_DayOfWeek"] = dates.dt.dayofweek
        X = X.drop(columns=["Date"])

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
        ("classifier", ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Test fraud prevalence:", round(y_test.mean(), 4))
    print("Extra Trees ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))
    print("Extra Trees PR-AUC:", round(
        average_precision_score(y_test, probabilities), 4
    ))
    print("Random-ranking PR-AUC baseline:", round(y_test.mean(), 4))


if __name__ == "__main__":
    main()