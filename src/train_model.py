from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.model_utils import prepare_features, get_models


# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "Data" / "processed" / "cleaned_fraud_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset path:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")


# =====================================================
# 2. LOAD AND VALIDATE DATA
# =====================================================

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

TARGET = "Fraud_Label"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Fraud_Label must contain only 0 and 1.")

print("\nFraud distribution:")
print(df[TARGET].value_counts())
print("\nFraud percentages:")
print((df[TARGET].value_counts(normalize=True) * 100).round(2))


# =====================================================
# 3. PREPARE FEATURES
# =====================================================

X, y, preprocessor = prepare_features(df)

print("\nInput features:", X.shape[1])
print("Target records:", len(y))

if y.nunique() != 2:
    raise ValueError("Both fraud labels 0 and 1 are required.")


# =====================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

# 60% training, 20% validation, 20% final testing.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))


# =====================================================
# 5. TRAIN BOTH MODELS
# =====================================================

trained_models = {}
baseline_results = []
validation_probabilities = {}

for model_name, classifier in get_models().items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    # Baseline predictions at the default threshold.
    test_pred = pipeline.predict(X_test)
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    baseline_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(
            y_test, test_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, test_pred, zero_division=0
        ),
        "PR_AUC": average_precision_score(y_test, test_prob),
    })

    trained_models[model_name] = pipeline

    # Validation probabilities are used for threshold selection.
    validation_probabilities[model_name] = (
        pipeline.predict_proba(X_val)[:, 1]
    )

    print("\nTest classification report:")
    print(classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))

    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_pred, labels=[0, 1]))


# =====================================================
# 6. SAVE BASELINE COMPARISON
# =====================================================

baseline_df = pd.DataFrame(baseline_results)

baseline_path = REPORT_DIR / "model_comparison.csv"
baseline_df.to_csv(baseline_path, index=False)

print("\nBASELINE MODEL COMPARISON")
print(baseline_df.to_string(index=False))


# =====================================================
# 7. TEST THRESHOLDS ON VALIDATION DATA
# =====================================================

print("\n" + "=" * 55)
print("VALIDATION THRESHOLD EXPERIMENT")
print("=" * 55)

threshold_results = []

thresholds = [0.20, 0.30, 0.40, 0.50]

for model_name, probabilities in validation_probabilities.items():

    for threshold in thresholds:

        val_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, val_pred, zero_division=0
        )
        recall = recall_score(
            y_val, val_pred, zero_division=0
        )
        f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        threshold_results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
        })

        print(
            f"{model_name} | Threshold={threshold:.2f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | F1={f1:.3f}"
        )

threshold_df = pd.DataFrame(threshold_results)

threshold_path = REPORT_DIR / "threshold_comparison.csv"
threshold_df.to_csv(threshold_path, index=False)


# =====================================================
# 8. SELECT MODEL AND THRESHOLD USING VALIDATION DATA
# =====================================================

# Select the model and threshold using validation F1,
# but reject thresholds that flag too many legitimate
# transactions. This is a practical constraint, not a
# universal fraud-detection rule.

MAX_FALSE_POSITIVE_RATE = 0.20

candidate_rows = []

for model_name, probabilities in validation_probabilities.items():

    legitimate_mask = (y_val == 0)

    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:

        val_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, val_pred, zero_division=0
        )
        recall = recall_score(
            y_val, val_pred, zero_division=0
        )
        f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        false_positive_rate = (
            ((val_pred == 1) & legitimate_mask).sum()
            / legitimate_mask.sum()
        )

        candidate_rows.append({
            "Model": model_name,
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "False_Positive_Rate": false_positive_rate,
        })

candidate_df = pd.DataFrame(candidate_rows)

candidate_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False,
)

eligible = candidate_df[
    candidate_df["False_Positive_Rate"] <= MAX_FALSE_POSITIVE_RATE
]

if eligible.empty:
    raise ValueError(
        "No threshold satisfies the false-positive-rate constraint. "
        "Inspect threshold_comparison.csv and revise the constraint."
    )

best_row = eligible.sort_values(
    ["F1_Score", "PR_AUC"] if "PR_AUC" in eligible.columns
    else ["F1_Score", "Recall"],
    ascending=False,
).iloc[0]

best_model_name = best_row["Model"]
best_threshold = float(best_row["Threshold"])
best_pipeline = trained_models[best_model_name]

print("\nSelected model:", best_model_name)
print("Selected threshold:", best_threshold)
print("Validation precision:", round(best_row["Precision"], 4))
print("Validation recall:", round(best_row["Recall"], 4))
print("Validation F1:", round(best_row["F1_Score"], 4))
print(
    "Validation false-positive rate:",
    round(best_row["False_Positive_Rate"], 4),
)

# =====================================================
# 9. FINAL EVALUATION ON UNSEEN TEST DATA
# =====================================================

print("\n" + "=" * 55)
print("FINAL TEST EVALUATION")
print("=" * 55)

final_prob = best_pipeline.predict_proba(X_test)[:, 1]
final_pred = (final_prob >= best_threshold).astype(int)

print("\nSelected model:", best_model_name)
print("Threshold:", best_threshold)

print("\nFinal precision:",
      round(precision_score(y_test, final_pred, zero_division=0), 4))
print("Final recall:",
      round(recall_score(y_test, final_pred, zero_division=0), 4))
print("Final F1-score:",
      round(f1_score(y_test, final_pred, zero_division=0), 4))
print("Final PR-AUC:",
      round(average_precision_score(y_test, final_prob), 4))

final_cm = confusion_matrix(y_test, final_pred, labels=[0, 1])
print("\nFinal confusion matrix:")
print(final_cm)

print("\nFinal classification report:")
print(classification_report(
    y_test,
    final_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))

# Save final confusion matrix.
tn, fp, fn, tp = final_cm.ravel()

pd.DataFrame([{
    "Model": best_model_name,
    "Threshold": best_threshold,
    "True_Negatives": int(tn),
    "False_Positives": int(fp),
    "False_Negatives": int(fn),
    "True_Positives": int(tp),
}]).to_csv(
    REPORT_DIR / "final_confusion_matrix.csv",
    index=False,
)

# Save final metrics.
pd.DataFrame([{
    "Model": best_model_name,
    "Threshold": best_threshold,
    "Precision": precision_score(
        y_test, final_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, final_pred, zero_division=0
    ),
    "F1_Score": f1_score(
        y_test, final_pred, zero_division=0
    ),
    "PR_AUC": average_precision_score(y_test, final_prob),
}]).to_csv(
    REPORT_DIR / "final_model_metrics.csv",
    index=False,
)


# =====================================================
# 10. SAVE MODEL AND THRESHOLD
# =====================================================

model_path = MODEL_DIR / "fraud_detection_pipeline.joblib"

joblib.dump({
    "pipeline": best_pipeline,
    "threshold": best_threshold,
    "model_name": best_model_name,
    "target_column": TARGET,
    "feature_columns": list(X.columns),
}, model_path)

(ROOT / "reports" / "feature_columns.txt").write_text(
    "\n".join(str(col) for col in X.columns),
    encoding="utf-8",
)

print("\n" + "=" * 55)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 55)
print("Selected model:", best_model_name)
print("Selected threshold:", best_threshold)
print("Saved model:", model_path)
print("Baseline comparison:", baseline_path)
print("Threshold comparison:", threshold_path)
print("Final metrics:", REPORT_DIR / "final_model_metrics.csv")