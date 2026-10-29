import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = "data/synthetic_fraud_dataset1.csv"
REPORT_PATH = "reports/engineered_feature_comparison.csv"
TARGET = "Fraud_Label"


def create_features(df):
    """Create features without using fraud labels or future transactions."""
    df = df.copy()

    # Parse the transaction date.
    df["_date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["_date"].isna().any():
        raise ValueError("Some dates could not be parsed.")

    # Basic date features.
    df["Transaction_Year"] = df["_date"].dt.year
    df["Transaction_Month"] = df["_date"].dt.month
    df["Transaction_Day"] = df["_date"].dt.day
    df["Transaction_DayOfWeek"] = df["_date"].dt.dayofweek
    df["Is_Weekend"] = (
        df["_date"].dt.dayofweek >= 5
    ).astype(int)

    # Transaction amount relative to account balance.
    balance = df["Account_Balance"].replace(0, np.nan)
    df["Amount_to_Balance_Ratio"] = (
        df["Transaction_Amount"] / balance
    )

    # Aggregate each user's transactions by date.
    # Transactions on the same date will not count as
    # historical transactions for one another.
    daily = (
        df.groupby(["User_ID", "_date"], as_index=False)
        .agg(
            Daily_User_Amount=("Transaction_Amount", "sum"),
            Daily_User_Count=("Transaction_ID", "size"),
        )
        .sort_values(["User_ID", "_date"])
    )

    # Calculate history using strictly earlier dates.
    daily["User_Prior_Transaction_Count"] = (
        daily.groupby("User_ID")["Daily_User_Count"]
        .transform(lambda s: s.cumsum().shift(1, fill_value=0))
    )

    daily["User_Prior_Amount_Total"] = (
        daily.groupby("User_ID")["Daily_User_Amount"]
        .transform(lambda s: s.cumsum().shift(1, fill_value=0))
    )

    daily["User_Prior_Avg_Amount"] = (
        daily["User_Prior_Amount_Total"]
        / daily["User_Prior_Transaction_Count"].replace(0, np.nan)
    )

    df = df.merge(
        daily[
            [
                "User_ID",
                "_date",
                "User_Prior_Transaction_Count",
                "User_Prior_Avg_Amount",
            ]
        ],
        on=["User_ID", "_date"],
        how="left",
        validate="many_to_one",
    )

    # First transactions have no prior average; the pipeline
    # will impute those missing values using training data.
    df = df.drop(
    columns=[
        TARGET,
        "Date",
        "_date",
        "Transaction_ID",
        "User_ID",
    ],
    errors="ignore",
)

    return df


def make_pipeline(X_train):
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(name, X_train, X_test, y_train, y_test):
    pipeline = make_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Experiment": name,
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(y_test, predictions, zero_division=0),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET]).copy()

    # Keep the original dataset intact.
    df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["_date"].isna().any():
        raise ValueError("Some dates could not be parsed.")

    df = df.sort_values("_date").reset_index(drop=True)

    # Chronological split: test on later dates.
    dates = np.sort(df["_date"].unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are required.")

    cutoff = dates[int(len(dates) * 0.8)]
    train_mask = df["_date"] < cutoff
    test_mask = ~train_mask

    train_raw = df.loc[train_mask].copy()
    test_raw = df.loc[test_mask].copy()

    y_train = train_raw[TARGET].astype(int)
    y_test = test_raw[TARGET].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Chronological split must contain both target classes "
            "in training and testing."
        )

    # Baseline features: current project date preprocessing,
    # without the new behavioral features.
    def baseline_features(frame):
        X = frame.drop(
            columns=[TARGET, "Transaction_ID", "User_ID"],
            errors="ignore",
        ).copy()

        dates_parsed = pd.to_datetime(X.pop("Date"), errors="coerce")
        X["Transaction_Year"] = dates_parsed.dt.year
        X["Transaction_Month"] = dates_parsed.dt.month
        X["Transaction_Day"] = dates_parsed.dt.day
        X["Transaction_DayOfWeek"] = dates_parsed.dt.dayofweek
        return X

    X_train_base = baseline_features(train_raw)
    X_test_base = baseline_features(test_raw)

    # Engineered features use history from strictly earlier dates.
    # Compute features across the complete timeline, without labels,
    # so test transactions can use available earlier transaction history.
    all_engineered = create_features(df)
    engineered_train_mask = train_mask.to_numpy()
    engineered_test_mask = test_mask.to_numpy()

    X_train_new = all_engineered.loc[engineered_train_mask].copy()
    X_test_new = all_engineered.loc[engineered_test_mask].copy()

    # Evaluate both experiments on exactly the same test transactions.
    results = [
        evaluate(
            "Baseline Random Forest",
            X_train_base, X_test_base, y_train, y_test
        ),
        evaluate(
            "Engineered Random Forest",
            X_train_new, X_test_new, y_train, y_test
        ),
    ]

    result_df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print("Dataset rows:", len(df))
    print("Training rows:", len(train_raw))
    print("Testing rows:", len(test_raw))
    print("Training ends before:", cutoff)
    print("\nFraud prevalence in test set:", round(y_test.mean(), 4))
    print("\nMODEL COMPARISON")
    print(result_df.to_string(index=False))
    print("\nSaved report:", REPORT_PATH)


if __name__ == "__main__":
    main()