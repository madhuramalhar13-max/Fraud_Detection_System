from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from src.model_utils import prepare_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "processed" / "cleaned_fraud_dataset.csv"

df = pd.read_csv(DATA_PATH)

X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

print("Training diagnostic model...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print("\nDIAGNOSTIC RESULTS")
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print(
    "PR-AUC:",
    round(average_precision_score(y_test, probabilities), 4)
)

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nTOP 15 FEATURES")
print(importance_df.head(15).to_string(index=False))

REPORT_PATH = ROOT / "reports" / "feature_importance.csv"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
importance_df.to_csv(REPORT_PATH, index=False)

print(f"\nFeature importance saved to: {REPORT_PATH}")