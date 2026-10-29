import json
from pathlib import Path

path = Path("notebooks/member2_fraud_model_evaluation.ipynb")

content = """# Financial Fraud Detection — Model Evaluation

## Member 2: Fraud Detection Model

## Objective
Evaluate fraud detection models, test engineered features, and tune the classification threshold.

## Dataset
- 50,000 transactions
- Target: `Fraud_Label`
- Fraud prevalence: approximately 32.13%

## Model Comparison
| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.4910 | 0.3175 |
| Random Forest | 0.5043 | 0.3246 |
| Gradient Boosting | 0.4944 | 0.3217 |

## Feature Engineering
Engineered Random Forest ROC-AUC: 0.5026; PR-AUC: 0.3264.
This is only a small improvement over the baseline.

## Threshold Tuning
Selected validation threshold: 0.25.
Test precision: 0.3228.
Test recall: 1.0000.
Test F1-score: 0.4881.

## Limitations
ROC-AUC near 0.50 indicates weak predictive discrimination.
The tested features show little relationship with the target.
The dataset's label-generation rule has not been identified.
These results do not establish production readiness.

## Reproducibility
See `reports/model_comparison.csv`,
`reports/engineered_feature_comparison.csv`, and
`reports/threshold_tuning_results.csv`.
"""

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.splitlines(keepends=True)
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Notebook repaired: {path}")
print(f"Size: {path.stat().st_size} bytes")