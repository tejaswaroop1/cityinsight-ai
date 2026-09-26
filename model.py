import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CIVICGUARD AI - REAL MACHINE LEARNING MODEL
# ============================================================
#
# Prediction task:
# Predict Incident Resolution Time (in Hours)
#
# Why this target?
# It is an actual field already present in cyber.csv.
#
# This is supervised regression.
# No artificial target variable is created.
# ============================================================


TARGET = "Incident Resolution Time (in Hours)"


FEATURES = [
    "Year",
    "Attack Type",
    "Target Industry",
    "Financial Loss (in Million $)",
    "Number of Affected Users",
    "Attack Source",
    "Security Vulnerability Type",
    "Defense Mechanism Used"
]


NUMERIC_FEATURES = [
    "Year",
    "Financial Loss (in Million $)",
    "Number of Affected Users"
]


CATEGORICAL_FEATURES = [
    "Attack Type",
    "Target Industry",
    "Attack Source",
    "Security Vulnerability Type",
    "Defense Mechanism Used"
]


def prepare_cyber_data(df):
    """
    Clean the cyber dataset for machine learning.
    """

    data = df.copy()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required cyber dataset columns: "
            + ", ".join(missing_columns)
        )

    # Convert numeric columns
    numeric_columns = NUMERIC_FEATURES + [TARGET]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove rows where target is unavailable
    data = data.dropna(
        subset=[TARGET]
    )

    # Remove invalid target values
    data = data[
        data[TARGET] >= 0
    ]

    # Clean categorical columns
    for column in CATEGORICAL_FEATURES:

        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    return data


def train_cyber_model(df, test_size=0.2, random_state=42):
    """
    Train a Random Forest regression model.

    Returns:
        model information dictionary
    """

    data = prepare_cyber_data(df)

    if len(data) < 10:
        raise ValueError(
            "Not enough valid cyber records for ML training. "
            "At least 10 records are required."
        )

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    regressor = RandomForestRegressor(
        n_estimators=250,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                regressor
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    r2 = r2_score(
        y_test,
        test_predictions
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = get_feature_importance(
        model
    )

    return {
        "model": model,

        "target": TARGET,

        "features": FEATURES,

        "training_records": len(X_train),

        "test_records": len(X_test),

        "total_records": len(data),

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "feature_importance": feature_importance,

        "X_test": X_test,

        "y_test": y_test,

        "predictions": test_predictions
    }


def get_feature_importance(model):
    """
    Extract feature importance from the trained
    Random Forest after one-hot encoding.
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    random_forest = model.named_steps[
        "model"
    ]

    # Get transformed feature names
    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance"
            ]
        )

    importances = (
        random_forest
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    # Clean sklearn prefixes
    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False
        )
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
    )

    # Sort
    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


def aggregate_feature_importance(
    importance_df
):
    """
    Combine one-hot encoded categories back
    into their original feature groups.

    Example:

    Attack Type_DDoS
    Attack Type_Malware
    Attack Type_Phishing

    become:

    Attack Type
    """

    if importance_df.empty:

        return importance_df

    result = []

    for _, row in importance_df.iterrows():

        feature = row["Feature"]
        importance = row["Importance"]

        original_feature = feature

        for column in CATEGORICAL_FEATURES:

            prefix = column + "_"

            if feature.startswith(prefix):

                original_feature = column

                break

        result.append(
            {
                "Feature": original_feature,
                "Importance": importance
            }
        )

    result_df = pd.DataFrame(result)

    result_df = (
        result_df
        .groupby("Feature")["Importance"]
        .sum()
        .reset_index()
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return result_df


def predict_resolution_time(
    model_info,
    incident_data
):
    """
    Predict resolution time for a new cyber incident.
    """

    model = model_info["model"]

    prediction = model.predict(
        incident_data[FEATURES]
    )

    return float(
        max(
            0,
            prediction[0]
        )
    )


def model_summary(model_info):
    """
    Return a clean summary for the Streamlit UI.
    """

    return {
        "Model": "Random Forest Regressor",

        "Target": TARGET,

        "Training Records":
            model_info["training_records"],

        "Test Records":
            model_info["test_records"],

        "MAE":
            model_info["mae"],

        "RMSE":
            model_info["rmse"],

        "R²":
            model_info["r2"]
    }