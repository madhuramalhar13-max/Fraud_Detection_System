from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.model_utils import prepare_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "processed" / "cleaned_fraud_dataset.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"

df = pd.read_csv(DATA_PATH)
X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    results.append({
        "Model": name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    })

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

results_df = pd.DataFrame(results).sort_values(
    "PR_AUC", ascending=False
)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORT_PATH, index=False)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {REPORT_PATH}")