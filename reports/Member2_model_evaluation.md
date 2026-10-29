# Member 2 — Fraud Detection Model Evaluation

## 1. Objective

The objective of Member 2 was to develop, evaluate, and improve a machine learning model for financial fraud detection using the provided synthetic transaction dataset.

## 2. Dataset Overview

- Dataset: `data/synthetic_fraud_dataset1.csv`
- Total transactions: 50,000
- Features and target columns: 14
- Legitimate transactions (Fraud_Label = 0): 33,933 (67.87%)
- Fraudulent transactions (Fraud_Label = 1): 16,067 (32.13%)

The target label is `Fraud_Label`. The original rule used to generate this label could not be identified in the repository.

## 3. Data Analysis and Feature Investigation

The following analyses were performed:
- Compared numeric feature distributions between legitimate and fraudulent transactions.
- Investigated fraud rates across transaction types, device types, locations, merchant categories, and card types.
- Examined transaction amount, account balance, card age, daily transaction count, and previous fraudulent activity.
- Checked the relationship between the available features and the target label.

Most observed fraud rates were close to the overall fraud prevalence. Numeric feature correlations with the target were also close to zero.

These findings suggest that the available features contain limited predictive information about the target label.

## 4. Preprocessing and Feature Engineering

The preprocessing pipeline includes:
- Excluding `Transaction_ID` and `User_ID` from ordinary model inputs.
- Extracting year, month, day, and day of week from transaction dates.
- Median imputation and standardization of numeric features.
- Most-frequent imputation and one-hot encoding of categorical features.

Additional features tested:
- `Amount_to_Balance_Ratio`
- `Is_Weekend`
- `User_Prior_Transaction_Count`
- `User_Prior_Avg_Amount`

Historical user features use transactions from earlier dates only. The dataset does not provide intraday transaction timestamps, so the order of transactions occurring on the same date cannot be established.

## 5. Model Comparison

Random Forest, Logistic Regression, and Gradient Boosting were compared using a stratified random split.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Random Forest | 0.5043 | 0.3246 |
| Gradient Boosting | 0.4944 | 0.3217 |
| Logistic Regression | 0.4910 | 0.3175 |

Random Forest ranked highest in this experiment. However, the results indicate weak discrimination between fraudulent and legitimate transactions.

## 6. Engineered Feature Experiment

A separate chronological experiment compared the baseline Random Forest with an engineered-feature version on the same test period.

| Metric | Baseline | Engineered |
|---|---:|---:|
| ROC-AUC | 0.4951 | 0.5026 |
| PR-AUC | 0.3225 | 0.3264 |
| Precision | 0.3538 | 0.3434 |
| Recall | 0.0609 | 0.0531 |
| F1-score | 0.1038 | 0.0920 |

The engineered model showed a small improvement in ROC-AUC and PR-AUC, but precision, recall, and F1-score decreased at threshold 0.5.

The improvement is limited and does not establish that the engineered model is suitable for operational fraud detection.

## 7. Classification Threshold Experiment

Threshold tuning used chronological training, validation, and test partitions. The threshold was selected using validation F1, subject to a target precision of at least 50%.

No tested threshold achieved 50% precision on validation data.

The fallback threshold was 0.25.

| Test metric | Threshold 0.50 | Threshold 0.25 |
|---|---:|---:|
| Precision | 0.3134 | 0.3228 |
| Recall | 0.0699 | 1.0000 |
| F1-score | 0.1142 | 0.4881 |
| ROC-AUC | 0.5048 | 0.5048 |
| PR-AUC | 0.3252 | 0.3252 |

At threshold 0.25, all test transactions were classified as fraudulent. The 100% recall therefore does not represent useful fraud discrimination. The threshold was not adopted as a production operating point.

## 8. Limitations

1. The original fraud-label generation rule was not located in the repository.
2. Fraud rates were similar across most observed feature categories.
3. Numeric feature correlations with the target were close to zero.
4. Tested models achieved ROC-AUC values close to 0.50.
5. The engineered-feature experiment produced only a small ranking improvement.
6. Threshold adjustment did not establish a useful precision-recall trade-off.
7. The dataset lacks intraday transaction timestamps, limiting transaction-velocity analysis.

## 9. Conclusion

The project successfully implemented preprocessing, feature engineering experiments, model comparison, and threshold evaluation.

Random Forest was the strongest model in the initial model comparison, but the available dataset produced weak predictive performance. Further improvement requires investigating how the fraud labels were generated and obtaining more informative fraud-related features or a better-validated dataset.

The current results should be treated as experimental findings rather than evidence of a production-ready fraud detection system.

## 10. Reproducibility

Run commands from the repository root:

```powershell
python -m src.analyze_features
python -m src.compare_models
python -m src.compare_engineered_features
python -m src.tune_threshold
python -m pip check
```

Generated results are stored in the `reports/` directory.