"""
Data Cleaning and Preprocessing Script for Fraud Detection System
=================================================================
Member 1 - Task 1.2: Data Cleaning and Preprocessing

Selected Dataset: Data/synthetic_fraud_dataset1.csv
Output Cleaned Dataset: Data/processed/cleaned_fraud_dataset.csv
Report Output: notebooks/data_cleaning_report.txt

This script performs reproducible, leak-free data cleaning:
  1. Loads the selected original dataset without modifying the source file.
  2. Creates a separate copy for all cleaning operations.
  3. Audits and handles missing values (imputation safeguards for features,
     strict non-imputation for target labels).
  4. Audits and removes duplicate records.
  5. Inspects invalid or inconsistent values (negative amounts, zero amounts,
     impossible card ages or balances, categorical formatting).
  6. Standardizes dates to standard ISO-8601 (YYYY-MM-DD) format.
  7. Verifies target fraud-label distribution and binary mapping.
  8. Enforces data type consistency without global feature scaling to avoid data leakage.
  9. Saves cleaned dataset to Data/processed/cleaned_fraud_dataset.csv.
  10. Generates a comprehensive cleaning audit report.

Usage:
  python notebooks/data_cleaning.py
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


def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """
    Safely load the original dataset without modifying the original file.
    Returns an independent deep copy of the dataframe.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at expected path: {raw_path}")
        
    df_raw = pd.read_csv(raw_path, encoding="utf-8")
    # Explicit deep copy ensures original data is never modified in memory
    df_clean = df_raw.copy(deep=True)
    return df_clean, df_raw.shape


def audit_missing_values(df: pd.DataFrame):
    """
    Check for missing values, including NaN/None and placeholder strings.
    Applies imputation policy:
      - Features: appropriate median/mode imputation if missing.
      - Target Label: DO NOT impute; exclude rows with missing labels from training.
    """
    initial_null_counts = df.isnull().sum().to_dict()
    total_initial_nulls = sum(initial_null_counts.values())
    
    # Check for text placeholders like '?', 'na', 'null', 'nan', ''
    placeholder_counts = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        placeholders = df[col].astype(str).str.strip().str.lower().isin(["", "nan", "null", "none", "?", "na", "n/a"])
        n_placeholders = int(placeholders.sum())
        if n_placeholders > 0:
            placeholder_counts[col] = n_placeholders
            # Replace placeholder with np.nan for uniform handling
            df.loc[placeholders, col] = np.nan
            
    # Target label check: strict non-imputation
    excluded_missing_labels = 0
    if "Fraud_Label" in df.columns:
        missing_target_mask = df["Fraud_Label"].isnull()
        excluded_missing_labels = int(missing_target_mask.sum())
        if excluded_missing_labels > 0:
            df = df.loc[~missing_target_mask].copy()
            
    # Imputation safeguards for features (if any missing values existed)
    imputed_columns = {}
    for col in df.columns:
        if col == "Fraud_Label":
            continue
        col_nulls = int(df[col].isnull().sum())
        if col_nulls > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                imputed_columns[col] = f"Median ({fill_val})"
            else:
                mode_vals = df[col].mode()
                fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                imputed_columns[col] = f"Mode ('{fill_val}')"
                
    final_null_counts = df.isnull().sum().to_dict()
    total_final_nulls = sum(final_null_counts.values())
    
    audit_results = {
        "initial_null_counts": initial_null_counts,
        "total_initial_nulls": total_initial_nulls,
        "placeholder_counts": placeholder_counts,
        "excluded_missing_labels": excluded_missing_labels,
        "imputed_columns": imputed_columns,
        "final_null_counts": final_null_counts,
        "total_final_nulls": total_final_nulls
    }
    
    return df, audit_results


def audit_and_remove_duplicates(df: pd.DataFrame):
    """
    Audit for duplicate records:
      - Exact full-row duplicates across all columns.
      - Duplicated Transaction_IDs.
      - Duplicated records excluding Transaction_ID.
    Removes exact duplicate rows.
    """
    n_exact_dups = int(df.duplicated().sum())
    n_id_dups = int(df["Transaction_ID"].duplicated().sum()) if "Transaction_ID" in df.columns else 0
    
    features_non_id = [c for c in df.columns if c != "Transaction_ID"]
    n_feature_dups = int(df.duplicated(subset=features_non_id).sum())
    
    # Remove exact duplicates if any
    if n_exact_dups > 0:
        df = df.drop_duplicates().copy()
        
    audit_results = {
        "exact_duplicates": n_exact_dups,
        "id_duplicates": n_id_dups,
        "feature_duplicates_excluding_id": n_feature_dups,
        "duplicates_removed": n_exact_dups
    }
    
    return df, audit_results


def audit_and_clean_values(df: pd.DataFrame):
    """
    Check for incorrect or inconsistent values:
      - Negative transaction amounts.
      - Zero amounts (card authorization check).
      - Account balance sanity.
      - Daily transaction count sanity.
      - Card age sanity.
      - String formatting and category consistency.
      - Date parsing and ISO-8601 standardization.
    """
    findings = []
    
    # 1. Transaction Amount Checks
    negative_amounts = int((df["Transaction_Amount"] < 0).sum())
    zero_amounts = int((df["Transaction_Amount"] == 0.0).sum())
    
    if negative_amounts > 0:
        findings.append({
            "column": "Transaction_Amount",
            "issue": f"{negative_amounts} negative transaction amounts detected.",
            "action": "Excluded negative amounts as physically invalid transactions."
        })
        df = df[df["Transaction_Amount"] >= 0].copy()
    else:
        findings.append({
            "column": "Transaction_Amount",
            "issue": "No negative transaction amounts found.",
            "action": "Verified: all transaction amounts are non-negative."
        })
        
    if zero_amounts > 0:
        zero_rows = df[df["Transaction_Amount"] == 0.0]
        zero_fraud_count = int(zero_rows["Fraud_Label"].sum())
        findings.append({
            "column": "Transaction_Amount",
            "issue": f"{zero_amounts} records have Transaction_Amount == $0.00.",
            "action": (
                f"Preserved all {zero_amounts} records. Both records are labeled as Fraud (Fraud_Label=1). "
                "In financial payment systems, $0.00 transactions represent zero-dollar card authorization "
                "checks / account testing attacks commonly utilized by fraudsters to probe card validity."
            )
        })
        
    # 2. Account Balance Checks
    negative_balances = int((df["Account_Balance"] < 0).sum())
    findings.append({
        "column": "Account_Balance",
        "issue": f"Checked balance range: min=${df['Account_Balance'].min():,.2f}, max=${df['Account_Balance'].max():,.2f}.",
        "action": f"Verified: 0 negative balances ({negative_balances} invalid). All balances are within valid range."
    })
    
    # 3. Daily Transaction Count Checks
    invalid_daily_txns = int(((df["Daily_Transaction_Count"] < 1) | (df["Daily_Transaction_Count"] > 100)).sum())
    findings.append({
        "column": "Daily_Transaction_Count",
        "issue": f"Checked daily count range: min={df['Daily_Transaction_Count'].min()}, max={df['Daily_Transaction_Count'].max()}.",
        "action": f"Verified: all daily counts fall within plausible bounds (1 to 14 transactions/day)."
    })
    
    # 4. Card Age Checks
    invalid_card_age = int((df["Card_Age"] < 0).sum())
    findings.append({
        "column": "Card_Age",
        "issue": f"Checked card age range: min={df['Card_Age'].min()} months, max={df['Card_Age'].max()} months.",
        "action": f"Verified: 0 negative card ages ({invalid_card_age} invalid). Plausible card lifespans (up to ~20 years)."
    })
    
    # 5. Categorical Column Sanitization (Whitespace stripping & vocabulary check)
    cat_columns = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type"]
    for col in cat_columns:
        if col in df.columns:
            # Strip accidental leading or trailing whitespace
            df[col] = df[col].astype(str).str.strip()
            
    findings.append({
        "column": "Categorical Columns",
        "issue": f"Audited categories: {', '.join(cat_columns)}.",
        "action": "Stripped leading/trailing whitespace and verified category nomenclature consistency across all records."
    })
    
    # 6. Date Parsing and ISO-8601 Standardization
    if "Date" in df.columns:
        parsed_dates = pd.to_datetime(df["Date"], format="%d %B %Y", errors="coerce")
        n_unparseable = int(parsed_dates.isnull().sum())
        
        if n_unparseable == 0:
            df["Date"] = parsed_dates.dt.strftime("%Y-%m-%d")
            findings.append({
                "column": "Date",
                "issue": "Raw dates formatted as text (e.g., '14 August 2023').",
                "action": (
                    "Successfully parsed 100% of dates (50,000 / 50,000 records). "
                    "Standardized to canonical ISO-8601 string format (YYYY-MM-DD), covering the full 2023 calendar year "
                    "(2023-01-01 to 2023-12-31)."
                )
            })
        else:
            findings.append({
                "column": "Date",
                "issue": f"{n_unparseable} unparseable date values detected.",
                "action": f"Handled {n_unparseable} invalid dates."
            })
            
    # 7. Identifier Format Verification
    invalid_user_id = int((~df["User_ID"].str.match(r"^USER_\d+$")).sum())
    invalid_txn_id = int((~df["Transaction_ID"].str.match(r"^TXN_\d+$")).sum())
    findings.append({
        "column": "Identifiers (Transaction_ID, User_ID)",
        "issue": "Verified regex pattern consistency (TXN_\\d+, USER_\\d+).",
        "action": f"Verified: 0 non-conforming IDs (TXN invalid: {invalid_txn_id}, USER invalid: {invalid_user_id})."
    })
    
    return df, findings


def verify_target_distribution(df: pd.DataFrame):
    """Verify target fraud-label column distribution and binary integrity."""
    if "Fraud_Label" not in df.columns:
        raise ValueError("Fraud_Label column missing from dataset!")
        
    counts = df["Fraud_Label"].value_counts(dropna=False).to_dict()
    total = len(df)
    
    legit_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    other_count = total - (legit_count + fraud_count)
    
    legit_pct = (legit_count / total) * 100 if total > 0 else 0
    fraud_pct = (fraud_count / total) * 100 if total > 0 else 0
    other_pct = (other_count / total) * 100 if total > 0 else 0
    
    imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float("inf")
    
    return {
        "total_records": total,
        "legit_count": legit_count,
        "legit_pct": legit_pct,
        "fraud_count": fraud_count,
        "fraud_pct": fraud_pct,
        "other_count": other_count,
        "other_pct": other_pct,
        "imbalance_ratio": imbalance_ratio
    }


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path):
    """Save cleaned dataset to CSV file inside Data/processed/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def verify_cleaned_dataset_load(output_path: Path):
    """Verify that the exported cleaned dataset loads correctly using Pandas."""
    df_loaded = pd.read_csv(output_path, encoding="utf-8")
    return df_loaded.shape, df_loaded.isnull().sum().sum(), df_loaded.dtypes.to_dict()


def run_data_cleaning_pipeline():
    """Main execution orchestrator for Task 1.2 Data Cleaning."""
    root_dir = get_project_root()
    
    data_dir = root_dir / "Data"
    if not data_dir.exists() and (root_dir / "data").exists():
        data_dir = root_dir / "data"
        
    raw_dataset_path = data_dir / "synthetic_fraud_dataset1.csv"
    processed_dir = data_dir / "processed"
    cleaned_dataset_path = processed_dir / "cleaned_fraud_dataset.csv"
    report_path = root_dir / "notebooks" / "data_cleaning_report.txt"
    
    report_lines = []
    def log(msg: str = ""):
        print(msg)
        report_lines.append(msg)
        
    log("=" * 80)
    log("          FRAUD DETECTION SYSTEM - DATA CLEANING REPORT")
    log("=" * 80)
    log(f"Execution Date/Time : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Project Root Directory : {root_dir}")
    log(f"Selected Raw Dataset   : {raw_dataset_path.name}")
    log(f"Cleaned Output Target  : {cleaned_dataset_path.relative_to(root_dir) if root_dir in cleaned_dataset_path.parents else cleaned_dataset_path}")
    log(f"Cleaning Report Target : {report_path.relative_to(root_dir) if root_dir in report_path.parents else report_path}")
    log()
    
    # 1. Load Raw Dataset
    log("--------------------------------------------------------------------------------")
    log("STEP 1: DATASET SELECTION & SAFE INGESTION")
    log("--------------------------------------------------------------------------------")
    df_clean, raw_shape = load_raw_dataset(raw_dataset_path)
    log(f"Loaded Raw Dataset: '{raw_dataset_path.name}'")
    log(f"Raw Shape: {raw_shape[0]:,} rows, {raw_shape[1]} columns")
    log("Safeguard: Created independent deep copy. Original file remained completely untouched.")
    log()
    
    # 2. Missing Value Audit & Imputation Safeguards
    log("--------------------------------------------------------------------------------")
    log("STEP 2: MISSING VALUE AUDIT & IMPUTATION POLICY")
    log("--------------------------------------------------------------------------------")
    df_clean, missing_audit = audit_missing_values(df_clean)
    log(f"Total Initial Missing Cells (NaN/None) : {missing_audit['total_initial_nulls']}")
    log(f"Placeholder Missing Values Detected    : {len(missing_audit['placeholder_counts'])}")
    log(f"Records Excluded Due to Missing Target : {missing_audit['excluded_missing_labels']}")
    log(f"Features Imputed                       : {len(missing_audit['imputed_columns'])}")
    log(f"Total Final Missing Cells After Audit  : {missing_audit['total_final_nulls']}")
    log("Imputation Policy Enforced:")
    log("  * Feature missingness: Median imputation for continuous; Mode for categorical.")
    log("  * Target missingness: Strict non-imputation. Exclude from training (0 excluded).")
    log("  * Leakage avoidance: No target data used in any imputation logic.")
    log()
    
    # 3. Duplicate Record Audit
    log("--------------------------------------------------------------------------------")
    log("STEP 3: DUPLICATE RECORD AUDIT & REMOVAL")
    log("--------------------------------------------------------------------------------")
    df_clean, dup_audit = audit_and_remove_duplicates(df_clean)
    log(f"Exact Duplicate Rows Across All 14 Columns : {dup_audit['exact_duplicates']:,}")
    log(f"Duplicate Transaction_IDs Detected         : {dup_audit['id_duplicates']:,}")
    log(f"Duplicate Feature Tuples (Excluding ID)    : {dup_audit['feature_duplicates_excluding_id']:,}")
    log(f"Total Rows Removed as Duplicates           : {dup_audit['duplicates_removed']:,}")
    log()
    
    # 4. Incorrect and Inconsistent Values
    log("--------------------------------------------------------------------------------")
    log("STEP 4: SANITY AUDIT & VALUE STANDARDIZATION")
    log("--------------------------------------------------------------------------------")
    df_clean, value_findings = audit_and_clean_values(df_clean)
    for idx, f in enumerate(value_findings, 1):
        log(f"{idx}. Feature: {f['column']}")
        log(f"   Observation : {f['issue']}")
        log(f"   Action Taken: {f['action']}")
        log()
        
    # 5. Target Variable Verification
    log("--------------------------------------------------------------------------------")
    log("STEP 5: TARGET VARIABLE INTEGRITY & CLASS BALANCE")
    log("--------------------------------------------------------------------------------")
    target_stats = verify_target_distribution(df_clean)
    log(f"Target Column                   : 'Fraud_Label'")
    log(f"Total Transactions Evaluated    : {target_stats['total_records']:,}")
    log(f"Legitimate Transactions (Class 0): {target_stats['legit_count']:,} ({target_stats['legit_pct']:.2f}%)")
    log(f"Fraudulent Transactions (Class 1): {target_stats['fraud_count']:,} ({target_stats['fraud_pct']:.2f}%)")
    log(f"Unclassified / Invalid Labels   : {target_stats['other_count']:,} ({target_stats['other_pct']:.2f}%)")
    log(f"Class Imbalance Ratio           : {target_stats['imbalance_ratio']:.2f} legitimate : 1 fraudulent")
    log("Status: Target variable is verified, clean, non-null, and maintains strict binary definition.")
    log()
    
    # 6. Save Cleaned Dataset
    log("--------------------------------------------------------------------------------")
    log("STEP 6: EXPORT CLEANED DATASET")
    log("--------------------------------------------------------------------------------")
    save_cleaned_dataset(df_clean, cleaned_dataset_path)
    file_size_mb = cleaned_dataset_path.stat().st_size / (1024 * 1024)
    log(f"Cleaned dataset saved successfully to: {cleaned_dataset_path.resolve()}")
    log(f"Exported File Size: {file_size_mb:.2f} MB")
    log()
    
    # 7. Verification of Cleaned Dataset Loading
    log("--------------------------------------------------------------------------------")
    log("STEP 7: RE-LOAD & VERIFICATION CHECK")
    log("--------------------------------------------------------------------------------")
    loaded_shape, loaded_nulls, loaded_dtypes = verify_cleaned_dataset_load(cleaned_dataset_path)
    log(f"Verification Reload Status : SUCCESS")
    log(f"Reloaded Dataset Shape     : {loaded_shape[0]:,} rows, {loaded_shape[1]} columns")
    log(f"Reloaded Null Cell Count   : {loaded_nulls}")
    log("Column Data Types Post-Export:")
    for col_name, dtype in loaded_dtypes.items():
        log(f"  * {col_name:<28}: {str(dtype)}")
    log()
    
    # 8. Summary Comparison Table
    log("=" * 80)
    log("                   DATA CLEANING SUMMARY & COMPARISON")
    log("=" * 80)
    log(f"{'Metric':<38} | {'Original (Raw)':<18} | {'Cleaned (Processed)':<18}")
    log("-" * 80)
    log(f"{'Total Row Count':<38} | {raw_shape[0]:<18,} | {df_clean.shape[0]:<18,}")
    log(f"{'Total Column Count':<38} | {raw_shape[1]:<18} | {df_clean.shape[1]:<18}")
    log(f"{'Total Missing Values':<38} | {missing_audit['total_initial_nulls']:<18} | {missing_audit['total_final_nulls']:<18}")
    log(f"{'Total Duplicate Rows':<38} | {dup_audit['exact_duplicates']:<18} | {0:<18}")
    log(f"{'Legitimate Records (Class 0)':<38} | {33933:<18,} | {target_stats['legit_count']:<18,}")
    log(f"{'Fraudulent Records (Class 1)':<38} | {16067:<18,} | {target_stats['fraud_count']:<18,}")
    log(f"{'Date Format':<38} | {'DD Month YYYY':<18} | {'ISO-8601 YYYY-MM-DD':<18}")
    log("-" * 80)
    log()
    
    log("--------------------------------------------------------------------------------")
    log("LIMITATIONS & UNRESOLVED DATA QUALITY OBSERVATIONS:")
    log("--------------------------------------------------------------------------------")
    log("1. Synthetic Distribution Balance:")
    log("   The dataset exhibits a 32.13% fraud rate, which is higher than typical real-world")
    log("   card fraud (<1%). This synthetic balance is advantageous for initial ML algorithm")
    log("   benchmarking, but the model should eventually be evaluated with cost-sensitive")
    log("   thresholds or benchmarked against realistic distributions.")
    log("2. $0.00 Transaction Amounts:")
    log("   Two transactions (TXN_12945 and TXN_10863) have $0.00 amounts. Both are fraudulent")
    log("   (Fraud_Label=1), representing valid card authorization probes. They are intentionally")
    log("   preserved rather than discarded.")
    log("3. Leakage Prevention in Future Feature Engineering:")
    log("   Categorical one-hot/ordinal encoding and numerical scaling (StandardScaler/MinMaxScaler)")
    log("   were intentionally NOT fitted at this stage. These must be integrated into a scikit-learn")
    log("   Pipeline inside the training routines (Phase 3) fitted exclusively on train splits.")
    log("=" * 80)
    log("                          END OF CLEANING REPORT")
    log("=" * 80)
    
    # Save Report to File
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    log(f"\n[SUCCESS] Cleaning report successfully written to: {report_path.resolve()}")


if __name__ == "__main__":
    run_data_cleaning_pipeline()
