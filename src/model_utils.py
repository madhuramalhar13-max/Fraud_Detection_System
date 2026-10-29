import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


TARGET_COLUMN = "Fraud_Label"

# Columns that identify a transaction or user,
# rather than describing transaction behaviour.
ID_COLUMNS = ["Transaction_ID", "User_ID"]


def prepare_features(df):
    """Separate input features from the fraud label."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' not found."
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    X = df.drop(columns=[TARGET_COLUMN] + [
        col for col in ID_COLUMNS if col in df.columns
    ])

    y = df[TARGET_COLUMN]

    # Convert dates into useful numerical features.
    if "Date" in X.columns:
        dates = pd.to_datetime(X["Date"], errors="coerce")
        X["Transaction_Year"] = dates.dt.year
        X["Transaction_Month"] = dates.dt.month
        X["Transaction_Day"] = dates.dt.day
        X["Transaction_DayOfWeek"] = dates.dt.dayofweek
        X = X.drop(columns=["Date"])

    # Separate numerical and categorical columns.
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return X, y, preprocessor


def get_models():
    """Return the two models required by the project."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }