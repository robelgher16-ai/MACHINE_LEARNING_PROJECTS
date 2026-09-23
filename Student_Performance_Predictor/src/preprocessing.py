
# ==========================================
# File: src/preprocessing.py
# Purpose: Create reusable ML preprocessing pipeline
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# Create Preprocessor
# ==========================================

def create_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Create the preprocessing pipeline for the
    Student Performance Predictor.

    Numerical features:
        1. Missing values -> median
        2. Scaling -> StandardScaler

    Categorical features:
        1. Missing values -> most frequent
        2. Encoding -> OneHotEncoder
        3. Unknown categories -> ignored

    Parameters
    ----------
    numerical_features : list[str]
        Names of numerical columns.

    categorical_features : list[str]
        Names of categorical columns.

    Returns
    -------
    ColumnTransformer
        Complete preprocessing pipeline.
    """

    # ==========================================
    # Numerical Pipeline
    # ==========================================

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # ==========================================
    # Categorical Pipeline
    # ==========================================

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])

    # ==========================================
    # Combine Pipelines
    # ==========================================

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor
