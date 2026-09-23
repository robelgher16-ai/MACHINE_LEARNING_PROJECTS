# ==========================================
# File: src/train.py
# Purpose: Train, compare, validate and save
#          the best Student Performance model
# ==========================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_cleaning import clean_dataset

from src.preprocessing import create_preprocessor


# ==========================================
# Suppress unnecessary warnings
# ==========================================

warnings.filterwarnings("ignore")


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5


# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Students_Performance.csv"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.pkl"
)


REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


METRICS_PATH = (
    REPORTS_DIR
    / "metrics.csv"
)


CV_RESULTS_PATH = (
    REPORTS_DIR
    / "cross_validation_results.csv"
)


METADATA_PATH = (
    REPORTS_DIR
    / "model_metadata.json"
)


# ==========================================
# Header
# ==========================================

print("=" * 75)
print("STUDENT PERFORMANCE PREDICTOR")
print("MODEL TRAINING PIPELINE")
print("=" * 75)


# ==========================================
# 1. Load Dataset
# ==========================================

print("\n[1/11] Loading dataset...")


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"""
Dataset not found.

Expected path:
{DATA_PATH}

Make sure the file exists:

data/raw/Students_Performance.csv
"""
    )


df_raw = pd.read_csv(
    DATA_PATH
)


print(
    f"Dataset shape: {df_raw.shape}"
)

print(
    f"Dataset path: {DATA_PATH}"
)


# ==========================================
# 2. Clean and Validate Dataset
# ==========================================

print(
    "\n[2/11] Cleaning and validating dataset..."
)


# IMPORTANT:
# Clean the dataset exactly once.

df_clean = clean_dataset(
    df_raw
)


print(
    "Dataset cleaning: OK"
)


print(
    f"Cleaned dataset shape: "
    f"{df_clean.shape}"
)


# ==========================================
# 3. Prepare Features and Target
# ==========================================

print(
    "\n[3/11] Preparing features and target..."
)


# Student_ID is an identifier.
# It must NOT be used as a predictive feature.

X = df_clean.drop(
    columns=[
        "Grade",
        "Student_ID"
    ]
)


y = df_clean["Grade"]


print(
    f"Features: {X.shape}"
)


print(
    f"Target:   {y.shape}"
)


# ==========================================
# 3.1 Validate Target
# ==========================================

print(
    "\n[3.1/11] Validating target..."
)


# Convert target to numeric.

y = pd.to_numeric(
    y,
    errors="raise"
)


# Convert target to integer.

y = y.astype(
    "int64"
)


# Validate missing values.

if y.isna().any():

    raise ValueError(
        "Target Grade contains missing values."
    )


# Validate numeric dtype.

if not pd.api.types.is_numeric_dtype(y):

    raise TypeError(
        "Target Grade must be numeric."
    )


# Validate target range.

if not y.between(
    0,
    7
).all():

    raise ValueError(
        "Target Grade contains values outside "
        "the expected range 0-7."
    )


print(
    "Target validation: OK"
)


print(
    "\nGrade distribution:"
)


print(
    y.value_counts()
    .sort_index()
)


# ==========================================
# 4. Identify Feature Types
# ==========================================

print(
    "\n[4/11] Identifying feature types..."
)


# Numerical columns

numerical_features = (
    X
    .select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    )
    .columns
    .tolist()
)


# Categorical columns

categorical_features = (
    X
    .select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )
    .columns
    .tolist()
)


print(
    "\nNumerical features:"
)


for feature in numerical_features:

    print(
        f"  - {feature}"
    )


print(
    "\nCategorical features:"
)


for feature in categorical_features:

    print(
        f"  - {feature}"
    )


# ==========================================
# 5. Train/Test Split
# ==========================================

print(
    "\n[5/11] Creating train/test split..."
)


# IMPORTANT:
#
# The test set remains untouched until
# final evaluation.
#
# Because the dataset is very small,
# we use stratification to preserve the
# grade distribution as much as possible.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)


print(
    f"Testing samples:  {len(X_test)}"
)


# ==========================================
# 6. Create Cross-Validation Strategy
# ==========================================

print(
    "\n[6/11] Creating cross-validation strategy..."
)


cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


print(
    f"Cross-validation folds: {CV_FOLDS}"
)


print(
    "Cross-validation strategy: OK"
)


# ==========================================
# 7. Define Models
# ==========================================

print(
    "\n[7/11] Creating machine learning models..."
)


models = {

    # --------------------------------------
    # Linear Models
    # --------------------------------------

    "Linear Regression":

        LinearRegression(),


    "Ridge Regression":

        Ridge(
            alpha=1.0
        ),


    "Lasso Regression":

        Lasso(
            alpha=0.01,
            max_iter=10000
        ),


    # --------------------------------------
    # Tree Models
    # --------------------------------------

    "Decision Tree":

        DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=4,
            min_samples_leaf=3
        ),


    "Random Forest":

        RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ),


    "Extra Trees":

        ExtraTreesRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


    # --------------------------------------
    # XGBoost
    # --------------------------------------

    "XGBoost":

        XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror"
        ),


    # --------------------------------------
    # LightGBM
    # --------------------------------------

    "LightGBM":

        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        )
}


print(
    f"Models to train: {len(models)}"
)


# ==========================================
# 8. Cross-Validation + Test Evaluation
# ==========================================

print(
    "\n[8/11] Training and evaluating models..."
)


print(
    "=" * 75
)


results = []


cv_results = []


trained_models = {}


for name, model in models.items():

    print(
        f"\nTraining: {name}"
    )


    # ======================================
    # Fresh preprocessing for every model
    # ======================================

    model_preprocessor = (
        create_preprocessor(
            numerical_features=
                numerical_features,

            categorical_features=
                categorical_features
        )
    )


    # ======================================
    # Complete ML Pipeline
    # ======================================

    pipeline = Pipeline([
        (
            "preprocessor",
            model_preprocessor
        ),

        (
            "model",
            model
        )
    ])


    # ======================================
    # Cross-Validation
    # ======================================

    print(
        "  Running cross-validation..."
    )


    # Negative MAE because sklearn scoring
    # maximizes scores.

    cv_mae_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )


    cv_rmse_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )


    cv_r2_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )


    cv_mae = -cv_mae_scores.mean()


    cv_mae_std = (
        cv_mae_scores.std()
    )


    cv_rmse = -cv_rmse_scores.mean()


    cv_rmse_std = (
        cv_rmse_scores.std()
    )


    cv_r2 = cv_r2_scores.mean()


    cv_r2_std = (
        cv_r2_scores.std()
    )


    # ======================================
    # Fit model on complete training set
    # ======================================

    pipeline.fit(
        X_train,
        y_train
    )


    # ======================================
    # Predict test set
    # ======================================

    predictions = pipeline.predict(
        X_test
    )


    # ======================================
    # Test Metrics
    # ======================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    mse = mean_squared_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mse
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    # ======================================
    # Store Test Results
    # ======================================

    results.append({

        "Model":
            name,

        "MAE":
            round(mae, 4),

        "MSE":
            round(mse, 4),

        "RMSE":
            round(rmse, 4),

        "R2":
            round(r2, 4),

        "CV_MAE":
            round(cv_mae, 4),

        "CV_MAE_STD":
            round(cv_mae_std, 4),

        "CV_RMSE":
            round(cv_rmse, 4),

        "CV_RMSE_STD":
            round(cv_rmse_std, 4),

        "CV_R2":
            round(cv_r2, 4),

        "CV_R2_STD":
            round(cv_r2_std, 4)
    })


    # ======================================
    # Store CV Results Separately
    # ======================================

    cv_results.append({

        "Model":
            name,

        "CV_MAE":
            round(cv_mae, 4),

        "CV_MAE_STD":
            round(cv_mae_std, 4),

        "CV_RMSE":
            round(cv_rmse, 4),

        "CV_RMSE_STD":
            round(cv_rmse_std, 4),

        "CV_R2":
            round(cv_r2, 4),

        "CV_R2_STD":
            round(cv_r2_std, 4)
    })


    # ======================================
    # Save trained pipeline
    # ======================================

    trained_models[name] = pipeline


    # ======================================
    # Display Results
    # ======================================

    print(
        f"  CV MAE   : "
        f"{cv_mae:.4f} ± {cv_mae_std:.4f}"
    )


    print(
        f"  CV RMSE  : "
        f"{cv_rmse:.4f} ± {cv_rmse_std:.4f}"
    )


    print(
        f"  CV R²    : "
        f"{cv_r2:.4f} ± {cv_r2_std:.4f}"
    )


    print(
        f"  Test MAE : "
        f"{mae:.4f}"
    )


    print(
        f"  Test RMSE: "
        f"{rmse:.4f}"
    )


    print(
        f"  Test R²  : "
        f"{r2:.4f}"
    )


# ==========================================
# 9. Model Comparison
# ==========================================

print("\n")


print(
    "=" * 75
)


print(
    "MODEL COMPARISON"
)


print(
    "=" * 75
)


results_df = pd.DataFrame(
    results
)


cv_results_df = pd.DataFrame(
    cv_results
)


# ==========================================
# Sort by Cross-Validation RMSE
# ==========================================

results_df = (
    results_df
    .sort_values(
        by="CV_RMSE",
        ascending=True
    )
    .reset_index(
        drop=True
    )
)


cv_results_df = (
    cv_results_df
    .sort_values(
        by="CV_RMSE",
        ascending=True
    )
    .reset_index(
        drop=True
    )
)


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 10. Select Best Model
# ==========================================

print(
    "\n[9/11] Selecting best model..."
)


# IMPORTANT:
#
# Select using cross-validation RMSE,
# NOT the test-set RMSE.
#
# This prevents the test set from
# influencing model selection.

best_model_name = (
    results_df
    .iloc[0]["Model"]
)


best_model = (
    trained_models[
        best_model_name
    ]
)


best_row = (
    results_df
    .iloc[0]
)


print(
    f"Selected model: "
    f"{best_model_name}"
)


print(
    f"CV RMSE: "
    f"{best_row['CV_RMSE']}"
)


print(
    f"Test RMSE: "
    f"{best_row['RMSE']}"
)


print(
    f"Test MAE: "
    f"{best_row['MAE']}"
)


print(
    f"Test R²: "
    f"{best_row['R2']}"
)


# ==========================================
# 11. Save Model and Reports
# ==========================================

print(
    "\n[10/11] Saving model and reports..."
)


# ==========================================
# Create Directories
# ==========================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# Save Complete Pipeline
# ==========================================

joblib.dump(
    best_model,
    MODEL_PATH
)


# ==========================================
# Save Metrics
# ==========================================

results_df.to_csv(
    METRICS_PATH,
    index=False
)


# ==========================================
# Save Cross-Validation Results
# ==========================================

cv_results_df.to_csv(
    CV_RESULTS_PATH,
    index=False
)


# ==========================================
# Metadata
# ==========================================

metadata = {

    "project":
        "Student Performance Predictor",


    "dataset": {

        "path":
            str(DATA_PATH),

        "rows":
            int(df_clean.shape[0]),

        "columns":
            int(df_clean.shape[1])
    },


    "features": {

        "count":
            int(X.shape[1]),

        "numerical":
            numerical_features,

        "categorical":
            categorical_features
    },


    "target": {

        "name":
            "Grade",

        "type":
            "ordinal_numeric",

        "range":
            [0, 7],

        "mapping": {

            "Fail": 0,

            "FF": 0,

            "DD": 1,

            "DC": 2,

            "CC": 3,

            "CB": 4,

            "BB": 5,

            "BA": 6,

            "AA": 7
        }
    },


    "training": {

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "training_samples":
            int(len(X_train)),

        "testing_samples":
            int(len(X_test)),

        "cross_validation_folds":
            CV_FOLDS
    },


    "best_model": {

        "name":
            best_model_name,

        "test_MAE":
            float(best_row["MAE"]),

        "test_MSE":
            float(best_row["MSE"]),

        "test_RMSE":
            float(best_row["RMSE"]),

        "test_R2":
            float(best_row["R2"]),

        "cv_MAE":
            float(best_row["CV_MAE"]),

        "cv_MAE_STD":
            float(best_row["CV_MAE_STD"]),

        "cv_RMSE":
            float(best_row["CV_RMSE"]),

        "cv_RMSE_STD":
            float(best_row["CV_RMSE_STD"]),

        "cv_R2":
            float(best_row["CV_R2"]),

        "cv_R2_STD":
            float(best_row["CV_R2_STD"])
    },


    "artifacts": {

        "model":
            str(MODEL_PATH),

        "metrics":
            str(METRICS_PATH),

        "cross_validation_results":
            str(CV_RESULTS_PATH),

        "metadata":
            str(METADATA_PATH)
    }
}


# ==========================================
# Save Metadata JSON
# ==========================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================
# Final Report
# ==========================================

print(
    "\n[11/11] Training complete"
)


print("\n")


print(
    "=" * 75
)


print(
    "TRAINING COMPLETE"
)


print(
    "=" * 75
)


print(
    f"Best Model : "
    f"{best_model_name}"
)


print(
    f"CV MAE    : "
    f"{best_row['CV_MAE']}"
)


print(
    f"CV RMSE   : "
    f"{best_row['CV_RMSE']}"
)


print(
    f"CV R²     : "
    f"{best_row['CV_R2']}"
)


print(
    f"Test MAE  : "
    f"{best_row['MAE']}"
)


print(
    f"Test MSE  : "
    f"{best_row['MSE']}"
)


print(
    f"Test RMSE : "
    f"{best_row['RMSE']}"
)


print(
    f"Test R²   : "
    f"{best_row['R2']}"
)


print(
    "\nSaved files:"
)


print(
    f"Model     : "
    f"{MODEL_PATH}"
)


print(
    f"Metrics   : "
    f"{METRICS_PATH}"
)


print(
    f"CV Results: "
    f"{CV_RESULTS_PATH}"
)


print(
    f"Metadata  : "
    f"{METADATA_PATH}"
)


print(
    "=" * 75
)