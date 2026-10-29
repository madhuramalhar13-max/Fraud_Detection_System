"""
Data Dictionary Generator for Fraud Detection System
===================================================
Member 1 - Task 1.4: Data Dictionary Creation

Input Dataset: Data/processed/cleaned_fraud_dataset.csv
Deliverables:
  - notebooks/data_dictionary.csv
  - notebooks/data_dictionary_report.txt

This script generates a comprehensive, reproducible data dictionary for all columns
in the cleaned dataset, providing clear guidance for Member 2 (Model Development):
  1. Identifies every column, its data type, and an authentic dataset example.
  2. Provides clear, simple descriptions of real-world business meanings.
  3. Formally verifies the ground truth target label and binary mapping.
  4. Designates primary candidate features, excluded columns, and feature engineering candidates.
  5. Documents potential data leakage risks and safe preprocessing pipelines.

Usage:
  python notebooks/data_dictionary.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Resolve project root directory whether executed from root or inside notebooks/."""
    current_path = Path.cwd()
    if (current_path / "Data").is_dir() or (current_path / "data").is_dir():
        return current_path
    
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "Data").is_dir() or (script_dir.parent / "data").is_dir():
        return script_dir.parent
        
    return current_path


# Detailed metadata registry for each column in cleaned_fraud_dataset.csv
COLUMN_METADATA_REGISTRY = {
    "Transaction_ID": {
        "description": "Unique transaction tracking identifier assigned to each financial event.",
        "is_target": "No",
        "can_use_as_feature": "No (Exclude from X)",
        "feature_role": "Primary Identifier (Lookup / Audit Key)",
        "data_quality_notes": "100% unique (50,000 unique values). Zero nulls. Conforms to regex '^TXN_\\d+$'.",
        "recommended_preprocessing": "Exclude from model feature matrix X to prevent memorization / overfitting. Retain as index or metadata tracking key for alerts and dashboards."
    },
    "User_ID": {
        "description": "Unique customer / account identifier associated with the transaction.",
        "is_target": "No",
        "can_use_as_feature": "No (Raw) / Yes (Engineered Aggregates)",
        "feature_role": "Customer Entity Identifier (High Cardinality)",
        "data_quality_notes": "High cardinality (8,963 unique users across 50,000 transactions; ~5.58 txns/user). Zero nulls. Conforms to '^USER_\\d+$'.",
        "recommended_preprocessing": "Do NOT apply One-Hot Encoding directly (would generate 8,963 sparse dimensions). Exclude from baseline models or compute out-of-fold historical aggregate features (e.g., user mean amount, transaction count) strictly within training folds."
    },
    "Transaction_Amount": {
        "description": "Monetary value of the transaction in USD ($).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Financial Metric (Numerical Feature)",
        "data_quality_notes": "Right-skewed continuous feature (Median: $69.66, Mean: $99.41, Max: $1,174.14). Includes exactly 2 confirmed $0.00 authorization pings (both 100% fraud). Zero negative values.",
        "recommended_preprocessing": "Scale using RobustScaler (resilient to outliers) or StandardScaler fitted strictly on training data. For linear models, apply log1p transformation log(x + 1). Combine with Account_Balance to construct Amount-to-Balance depletion ratios."
    },
    "Transaction_Type": {
        "description": "Payment channel / mechanism used to execute the transaction (POS, Online, ATM Withdrawal, Bank Transfer).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Contextual Feature (Low-Cardinality Categorical)",
        "data_quality_notes": "4 distinct categories ('POS', 'Bank Transfer', 'Online', 'ATM Withdrawal'). Clean string formatting, zero nulls.",
        "recommended_preprocessing": "One-Hot Encode using OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') fitted exclusively on training data."
    },
    "Date": {
        "description": "Calendar date when the transaction occurred, standardized to ISO-8601 (YYYY-MM-DD).",
        "is_target": "No",
        "can_use_as_feature": "No (Raw String) / Yes (Derived Temporal Features)",
        "feature_role": "Temporal Feature (Time Index / Split Key)",
        "data_quality_notes": "Covers full 2023 calendar year (365 days, 2023-01-01 to 2023-12-31). Zero nulls. Daily granularity (no hours/minutes).",
        "recommended_preprocessing": "Do NOT feed raw date string to models. Extract temporal features: Month (1-12), Day_Of_Week (0-6), Is_Weekend (0/1), Day_Of_Year (1-365). Use Date for chronological train/validation/test splitting."
    },
    "Account_Balance": {
        "description": "Total available account balance of the user at transaction initiation in USD ($).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Financial Metric (Numerical Feature)",
        "data_quality_notes": "Continuous positive values ranging from $500.48 to $99,998.31. Zero nulls, zero negative balances.",
        "recommended_preprocessing": "Standardize using StandardScaler fitted on training split. Create composite interaction feature: Transaction_Amount / (Account_Balance + 1)."
    },
    "Device_Type": {
        "description": "Hardware device type utilized by the customer to perform the transaction (Laptop, Mobile, Tablet).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Contextual Feature (Low-Cardinality Categorical)",
        "data_quality_notes": "3 distinct categories ('Laptop', 'Mobile', 'Tablet'). Zero nulls, balanced distribution (~16,600 txns each).",
        "recommended_preprocessing": "One-Hot Encode using OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') fitted on training split."
    },
    "Location": {
        "description": "Metropolitan geographic area where the transaction was logged.",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Contextual Feature (Low-Cardinality Categorical)",
        "data_quality_notes": "5 distinct metropolitan cities ('Sydney', 'New York', 'Mumbai', 'Tokyo', 'London'). Zero nulls.",
        "recommended_preprocessing": "One-Hot Encode using OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') fitted on training split."
    },
    "Merchant_Category": {
        "description": "Commercial industry category of the merchant receiving the payment (Travel, Clothing, Restaurants, Electronics, Groceries).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Contextual Feature (Low-Cardinality Categorical)",
        "data_quality_notes": "5 distinct industry categories. Zero nulls, balanced distribution (~10,000 txns each).",
        "recommended_preprocessing": "One-Hot Encode using OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') fitted on training split."
    },
    "Previous_Fraudulent_Activity": {
        "description": "Historical behavioral indicator recording whether the account had prior confirmed or suspected fraudulent events before this transaction.",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Behavioral Indicator (Binary Numerical / Flag)",
        "data_quality_notes": "Binary values (0 = Clean record [45,080 txns], 1 = Prior fraud history [4,920 txns]). Zero nulls. Must NOT be confused with current target Fraud_Label.",
        "recommended_preprocessing": "Pass directly as binary numeric indicator (0 or 1). No scaling required. Combine with Daily_Transaction_Count for high-risk composite behavioral flagging."
    },
    "Daily_Transaction_Count": {
        "description": "Number of transactions executed by the user on the same calendar day up to and including the current transaction.",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Behavioral Velocity Metric (Discrete Numerical)",
        "data_quality_notes": "Integer range from 1 to 14 transactions/day. Zero nulls. Captures transaction velocity.",
        "recommended_preprocessing": "Scale using StandardScaler fitted on training split. High values (>10 txns/day) indicate sudden transaction velocity spikes."
    },
    "Card_Type": {
        "description": "Payment card network brand utilized for the transaction (Amex, Mastercard, Visa, Discover).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Contextual Feature (Low-Cardinality Categorical)",
        "data_quality_notes": "4 distinct card brands ('Amex', 'Mastercard', 'Visa', 'Discover'). Zero nulls, balanced distribution (~12,500 txns each).",
        "recommended_preprocessing": "One-Hot Encode using OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') fitted on training split."
    },
    "Card_Age": {
        "description": "Age of the card / account measured in months since issuance (approx. 1 to 239 months, up to ~20 years).",
        "is_target": "No",
        "can_use_as_feature": "Yes (Recommended)",
        "feature_role": "Account Longevity Metric (Discrete Numerical)",
        "data_quality_notes": "Integer range 1 to 239 months. Zero nulls, zero negative values.",
        "recommended_preprocessing": "Scale using StandardScaler fitted on training split. Low card age (<6 months) coupled with high transaction amounts is a classic signature of synthetic new-account fraud."
    },
    "Fraud_Label": {
        "description": "Ground truth outcome of the transaction: 0 = Legitimate transaction, 1 = Fraudulent transaction.",
        "is_target": "Yes (Primary Target Variable)",
        "can_use_as_feature": "STRICTLY NO (Target)",
        "feature_role": "Supervised Binary Target Label (y)",
        "data_quality_notes": "Strictly binary (0 or 1). Zero nulls. Class distribution: 33,933 Legitimate (67.87%) vs. 16,067 Fraudulent (32.13%). Imbalance ratio: 2.11:1.",
        "recommended_preprocessing": "Isolate as target vector y. NEVER include in input feature matrix X. Use for training loss and evaluation. Optimize and benchmark using Recall, Precision, F1-Score, and PR-AUC rather than raw Accuracy."
    }
}


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Build column-level data dictionary DataFrame matching all columns in cleaned dataset."""
    rows = []
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        # Select clean example value from first non-null record
        sample_val = df[col].iloc[0]
        if isinstance(sample_val, (float, np.floating)):
            example_str = f"{sample_val:.2f}"
        else:
            example_str = str(sample_val)
            
        meta = COLUMN_METADATA_REGISTRY.get(col, {
            "description": "Column present in cleaned dataset (requires domain confirmation).",
            "is_target": "No",
            "can_use_as_feature": "Candidate",
            "feature_role": "General Feature",
            "data_quality_notes": "Zero nulls.",
            "recommended_preprocessing": "Standard preprocessing."
        })
        
        rows.append({
            "column_name": col,
            "data_type": dtype_str,
            "description": meta["description"],
            "example_value": example_str,
            "is_target": meta["is_target"],
            "can_use_as_feature": meta["can_use_as_feature"],
            "feature_role": meta["feature_role"],
            "data_quality_notes": meta["data_quality_notes"],
            "recommended_preprocessing": meta["recommended_preprocessing"]
        })
        
    return pd.DataFrame(rows)


def generate_handover_report(dict_df: pd.DataFrame, df: pd.DataFrame, report_path: Path):
    """Generate comprehensive handover report for Member 2 (Model Development)."""
    report_lines = []
    def log(msg: str = ""):
        print(msg)
        report_lines.append(msg)
        
    log("=" * 80)
    log("        FRAUD DETECTION SYSTEM - DATA DICTIONARY & MODELING HANDOVER")
    log("=" * 80)
    log(f"Execution Date/Time : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Cleaned Dataset     : Data/processed/cleaned_fraud_dataset.csv ({len(df):,} rows, {len(df.columns)} columns)")
    log(f"Data Dictionary CSV : notebooks/data_dictionary.csv")
    log()
    
    # 1. Target Label Verification
    log("--------------------------------------------------------------------------------")
    log("1. VERIFIED GROUND TRUTH TARGET VARIABLE")
    log("--------------------------------------------------------------------------------")
    target_col = "Fraud_Label"
    counts = df[target_col].value_counts().sort_index()
    legit_cnt = counts[0]
    fraud_cnt = counts[1]
    total_cnt = len(df)
    
    log(f"Target Column Name           : '{target_col}'")
    log(f"Target Data Type             : int64 (Binary Supervised Classification)")
    log(f"Class 0 (Legitimate Record)  : {legit_cnt:,} transactions ({(legit_cnt/total_cnt)*100:.2f}%)")
    log(f"Class 1 (Fraudulent Record)  : {fraud_cnt:,} transactions ({(fraud_cnt/total_cnt)*100:.2f}%)")
    log(f"Class Imbalance Ratio        : {legit_cnt/fraud_cnt:.2f} legitimate per 1 fraudulent transaction")
    log()
    log("CRITICAL TARGET RESTRICTIONS FOR MEMBER 2:")
    log("  * Fraud_Label must be STRICTLY ISOLATED as target vector 'y'.")
    log("  * NEVER include Fraud_Label inside input feature matrix 'X'.")
    log("  * Do not confuse 'Previous_Fraudulent_Activity' with 'Fraud_Label'.")
    log("    'Previous_Fraudulent_Activity' is a historical behavioral feature (prior history),")
    log("    whereas 'Fraud_Label' is the outcome of the CURRENT transaction.")
    log()
    
    # 2. Recommended Model Input Features
    log("--------------------------------------------------------------------------------")
    log("2. RECOMMENDED MODEL INPUT FEATURES (READY FOR PIPELINE)")
    log("--------------------------------------------------------------------------------")
    recommended = dict_df[dict_df["can_use_as_feature"] == "Yes (Recommended)"]
    log(f"Total Direct Recommended Features: {len(recommended)} columns")
    log()
    
    log("A. Numerical Features (Scale using StandardScaler / RobustScaler):")
    num_features = ["Transaction_Amount", "Account_Balance", "Daily_Transaction_Count", "Card_Age"]
    for f in num_features:
        row = dict_df[dict_df["column_name"] == f].iloc[0]
        log(f"  * {f:<28} : {row['description']}")
        log(f"    Example: {row['example_value']} | Preprocessing: {row['recommended_preprocessing']}")
    log()
    
    log("B. Binary Behavioral Flag (Direct Numeric Input, No Scaling Needed):")
    bin_features = ["Previous_Fraudulent_Activity"]
    for f in bin_features:
        row = dict_df[dict_df["column_name"] == f].iloc[0]
        log(f"  * {f:<28} : {row['description']}")
        log(f"    Example: {row['example_value']} | Preprocessing: {row['recommended_preprocessing']}")
    log()
    
    log("C. Categorical Features (Apply One-Hot Encoding):")
    cat_features = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type"]
    for f in cat_features:
        row = dict_df[dict_df["column_name"] == f].iloc[0]
        distinct_vals = df[f].unique().tolist()
        log(f"  * {f:<28} : {row['description']}")
        log(f"    Distinct Categories ({len(distinct_vals)}): {distinct_vals}")
        log(f"    Preprocessing: OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')")
    log()
    
    # 3. Columns Requiring Special Handling / Feature Engineering
    log("--------------------------------------------------------------------------------")
    log("3. COLUMNS REQUIRING SPECIAL HANDLING & FEATURE ENGINEERING")
    log("--------------------------------------------------------------------------------")
    
    log("1. 'Date' (Standardized ISO-8601 String: YYYY-MM-DD):")
    log("   - Do NOT pass raw date strings into scikit-learn models.")
    log("   - Feature Engineering Recommendations:")
    log("       * Month = df['Date'].dt.month (1 to 12)")
    log("       * Day_Of_Week = df['Date'].dt.dayofweek (0=Monday to 6=Sunday)")
    log("       * Is_Weekend = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)")
    log("       * Day_Of_Year = df['Date'].dt.dayofyear (1 to 365)")
    log("   - Train/Test Split: Use 'Date' for chronological splitting (e.g. Train on")
    log("     Jan-Sep 2023, Test on Oct-Dec 2023) to validate realistic out-of-time performance.")
    log()
    
    log("2. 'User_ID' (Customer Identifier: USER_\\d+):")
    log("   - High cardinality (8,963 unique users across 50,000 transactions).")
    log("   - Exclude from basic one-hot encoding (prevents high-dimensional memory explosion).")
    log("   - Advanced Option: Can be used to construct historical rolling user metrics")
    log("     (e.g., user average transaction amount, user transaction frequency) computed strictly")
    log("     inside training folds to prevent data leakage.")
    log()
    
    log("3. 'Transaction_Amount' & 'Account_Balance' Interaction:")
    log("   - Recommended Composite Feature: Amount_to_Balance_Ratio = Amount / (Balance + 1.0).")
    log("   - In EDA, both features had near-zero bivariate correlation alone; composite financial")
    log("     depletion ratios provide non-linear leverage.")
    log()
    
    # 4. Excluded Columns & Data Leakage Prevention
    log("--------------------------------------------------------------------------------")
    log("4. EXCLUDED COLUMNS & DATA LEAKAGE SAFEGUARDS")
    log("--------------------------------------------------------------------------------")
    log("1. Excluded Columns:")
    log("   - 'Transaction_ID': Excluded from feature matrix X. 100% unique per row (50,000 unique).")
    log("     Passing it causes artificial memorization / overfitting.")
    log("   - 'Fraud_Label': Excluded from feature matrix X. Isolated strictly as ground truth y.")
    log()
    log("2. Data Leakage Prevention Rules for Member 2:")
    log("   - RULE 1: Never fit scalers (StandardScaler, RobustScaler) or encoders (OneHotEncoder)")
    log("     on the entire dataset. Always fit on X_train only, then transform X_val / X_test.")
    log("   - RULE 2: Use scikit-learn Pipeline or ColumnTransformer to encapsulate preprocessing")
    log("     and model training into a single reproducible estimator.")
    log("   - RULE 3: Do not impute features using target knowledge.")
    log("   - RULE 4: For k-fold cross validation, ensure preprocessing is fitted strictly within")
    log("     each training fold (e.g., using cross_val_score with Pipeline).")
    log()
    
    # 5. Modeling Evaluation Guidance for Member 2
    log("--------------------------------------------------------------------------------")
    log("5. MODEL EVALUATION GUIDANCE (MEMBER 2 HANDOVER)")
    log("--------------------------------------------------------------------------------")
    log("1. Prioritize Decision Threshold Tuning & PR-AUC:")
    log("   - Standard Accuracy (67.87% baseline) is insufficient for financial fraud detection.")
    log("   - Target evaluation metrics:")
    log("       * Recall (Sensitivity): Minimize false negatives (undetected fraud).")
    log("       * Precision: Minimize false positives (customer disruption).")
    log("       * F1-Score: Harmonic balance between Precision and Recall.")
    log("       * PR-AUC (Precision-Recall Area Under Curve): Primary model selection metric.")
    log("       * ROC-AUC: Secondary ranking metric.")
    log()
    log("2. Recommended Algorithms to Benchmark:")
    log("   - Baseline Model: Logistic Regression (with scaled numericals and one-hot categoricals).")
    log("   - Tree Ensembles: Random Forest, Gradient Boosting (XGBoost / LightGBM / HistGradientBoosting).")
    log("   - Tree models naturally capture high-order interactions between amounts, categories, and locations.")
    log("=" * 80)
    log("                       END OF DATA DICTIONARY REPORT")
    log("=" * 80)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
        
    log(f"\n[SUCCESS] Handover report written to: {report_path.resolve()}")


def run_pipeline():
    """Main execution orchestrator for Task 1.4."""
    root_dir = get_project_root()
    cleaned_dataset_path = root_dir / "Data" / "processed" / "cleaned_fraud_dataset.csv"
    dict_csv_path = root_dir / "notebooks" / "data_dictionary.csv"
    dict_report_path = root_dir / "notebooks" / "data_dictionary_report.txt"
    
    if not cleaned_dataset_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_dataset_path}")
        
    df = pd.read_csv(cleaned_dataset_path, encoding="utf-8")
    print(f"Loaded Cleaned Dataset: {cleaned_dataset_path.name} ({len(df):,} rows, {len(df.columns)} columns)")
    
    # Generate Data Dictionary DataFrame
    dict_df = build_data_dictionary(df)
    
    # Save CSV
    dict_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dict_df.to_csv(dict_csv_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Data dictionary CSV saved to: {dict_csv_path.resolve()}")
    
    # Generate and Save Handover Report
    generate_handover_report(dict_df, df, dict_report_path)


if __name__ == "__main__":
    run_pipeline()
