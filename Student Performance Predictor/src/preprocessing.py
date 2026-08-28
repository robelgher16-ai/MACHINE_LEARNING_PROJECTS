# ==========================================
# File: src/preprocessing.py
# Purpose: Create preprocessing pipeline
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(numerical_features, categorical_features):
    """
    Create a complete preprocessing pipeline.

    Parameters
    ----------
    numerical_features : list
        Numerical column names.

    categorical_features : list
        Categorical column names.

    Returns
    -------
    ColumnTransformer
        Ready-to-use preprocessing pipeline.
    """

    # Numerical preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor