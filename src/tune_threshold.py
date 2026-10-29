import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.compare_engineered_features import create_features, TARGET

DATA_PATH = "data/synthetic_fraud_dataset1.csv"
REPORT_PATH = "reports/threshold_tuning_results.csv"


def make_pipeline(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
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


def evaluate_threshold(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Threshold": round(float(threshold), 3),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
    }


def main():
    # 1. Load and sort data chronologically.
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET]).copy()
    df["_sort_date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["_sort_date"].isna().any():
        raise ValueError("Some transaction dates could not be parsed.")

    df = df.sort_values("_sort_date").reset_index(drop=True)

    dates = np.sort(df["_sort_date"].unique())

    if len(dates) < 3:
        raise ValueError(
            "Not enough distinct dates for chronological splitting."
        )

    # 2. Split into 60% training, 20% validation, 20% testing.
    cut1 = dates[int(len(dates) * 0.60)]
    cut2 = dates[int(len(dates) * 0.80)]

    train_mask = df["_sort_date"] < cut1
    val_mask = (
        (df["_sort_date"] >= cut1)
        & (df["_sort_date"] < cut2)
    )
    test_mask = df["_sort_date"] >= cut2

    y = df[TARGET].astype(int).reset_index(drop=True)

    # 3. Generate features. The function must exclude Fraud_Label.
    feature_source = df.drop(columns=["_sort_date"])
    X = create_features(feature_source).reset_index(drop=True)

    # Explicit leakage protection.
    if TARGET in X.columns:
        raise ValueError(
            f"Target leakage detected: {TARGET} is in the features."
        )

    X_train = X.loc[train_mask].copy()
    X_val = X.loc[val_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_val = y.loc[val_mask].copy()
    y_test = y.loc[test_mask].copy()

    for name, labels in [
        ("training", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:
        if labels.nunique() < 2:
            raise ValueError(
                f"The {name} split does not contain both classes."
            )

    # 4. Train using training data only.
    model = make_pipeline(X_train)
    model.fit(X_train, y_train)

    # 5. Obtain validation and test probabilities.
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # 6. Select threshold using validation data only.
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)

    validation_results = pd.DataFrame([
        evaluate_threshold(y_val, val_prob, threshold)
        for threshold in thresholds
    ])

    eligible = validation_results[
        validation_results["Precision"] >= 0.50
    ]

    if eligible.empty:
        print(
            "\nWARNING: No tested threshold achieved "
            "at least 50% precision on validation data."
        )

        # Fallback: highest validation F1.
        best_row = validation_results.loc[
            validation_results["F1"].idxmax()
        ]
    else:
        # Highest recall, then F1, among thresholds meeting
        # the validation precision requirement.
        best_row = eligible.sort_values(
            ["Recall", "F1", "Precision"],
            ascending=False,
        ).iloc[0]

    best_threshold = float(best_row["Threshold"])

    print("\nDATA SPLIT")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Testing rows:", len(X_test))
    print("Training ends before:", cut1)
    print("Validation ends before:", cut2)
    print("Validation fraud prevalence:", round(y_val.mean(), 4))
    print("Test fraud prevalence:", round(y_test.mean(), 4))

    print("\nSELECTED VALIDATION THRESHOLD")
    print(best_row.to_string())

    print("\nTOP VALIDATION THRESHOLDS")
    if eligible.empty:
        print("No threshold met the 50% precision requirement.")
    else:
        print(
            eligible.sort_values(
                ["Recall", "F1"], ascending=False
            ).head(10).to_string(index=False)
        )

    # 7. Evaluate the default and selected thresholds on test data.
    results = []

    for name, threshold in [
        ("Default threshold", 0.50),
        ("Validation-selected threshold", best_threshold),
    ]:
        metrics = evaluate_threshold(y_test, test_prob, threshold)

        metrics["Experiment"] = name
        metrics["ROC_AUC"] = roc_auc_score(y_test, test_prob)
        metrics["PR_AUC"] = average_precision_score(y_test, test_prob)

        results.append(metrics)

    results_df = pd.DataFrame(results)[[
        "Experiment",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
    ]]

    # 8. Save reports.
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)

    print("\nFINAL TEST RESULTS")
    print(results_df.to_string(index=False))
    print("\nSaved report:", REPORT_PATH)


if __name__ == "__main__":
    main()