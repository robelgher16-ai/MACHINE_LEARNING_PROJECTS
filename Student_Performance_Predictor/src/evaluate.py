# ==========================================
# File: src/evaluate.py
# Purpose: Evaluate the trained ML model
# ==========================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_cleaning import prepare_features


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    / "evaluation_metrics.json"
)

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "predictions.csv"
)

IMPORTANCE_PATH = (
    REPORTS_DIR
    / "feature_importance.csv"
)


# ==========================================
# Header
# ==========================================

print("=" * 70)
print("STUDENT PERFORMANCE PREDICTOR")
print("MODEL EVALUATION")
print("=" * 70)


# ==========================================
# 1. Validate Files
# ==========================================

print("\n[1/8] Checking project files...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}\n"
        "Run training first with:\n"
        "python -m src.train"
    )

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Dataset: OK")
print("Model:   OK")


# ==========================================
# 2. Load Dataset
# ==========================================

print("\n[2/8] Loading and cleaning dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Original dataset shape: {df.shape}"
)

X, y = prepare_features(df)

print(
    f"Features shape: {X.shape}"
)

print(
    f"Target shape:   {y.shape}"
)


# ==========================================
# 3. Validate Target
# ==========================================

print("\n[3/8] Validating target...")

if not pd.api.types.is_numeric_dtype(y):
    y = pd.to_numeric(
        y,
        errors="raise"
    )

y = y.astype(int)

if y.isna().any():
    raise ValueError(
        "Target contains missing values."
    )

print("Target: OK")

print("\nGrade distribution:")
print(
    y.value_counts()
    .sort_index()
)


# ==========================================
# 4. Create Same Test Set
# ==========================================

print("\n[4/8] Creating evaluation dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)


# ==========================================
# 5. Load Trained Model
# ==========================================

print("\n[5/8] Loading trained model...")

model = joblib.load(
    MODEL_PATH
)

print(
    f"Loaded model: {MODEL_PATH}"
)

print(
    f"Model type: "
    f"{type(model.named_steps['model']).__name__}"
)


# ==========================================
# 6. Generate Predictions
# ==========================================

print("\n[6/8] Generating predictions...")

predictions = model.predict(
    X_test
)

print("Prediction: OK")


# ==========================================
# 7. Calculate Metrics
# ==========================================

print("\n[7/8] Calculating evaluation metrics...")

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# ==========================================
# Display Metrics
# ==========================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"MSE  : {mse:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ==========================================
# Grade Conversion
# ==========================================

grade_names = {
    0: "Fail",
    1: "DD",
    2: "DC",
    3: "CC",
    4: "CB",
    5: "BB",
    6: "BA",
    7: "AA"
}


# ==========================================
# Create Prediction Table
# ==========================================

comparison = pd.DataFrame({
    "Actual_Grade": y_test.values,
    "Predicted_Grade": predictions
})

comparison["Predicted_Grade"] = (
    comparison["Predicted_Grade"]
    .clip(0, 7)
)

comparison["Predicted_Grade_Rounded"] = (
    comparison["Predicted_Grade"]
    .round()
    .astype(int)
)

comparison["Actual_Label"] = (
    comparison["Actual_Grade"]
    .map(grade_names)
)

comparison["Predicted_Label"] = (
    comparison["Predicted_Grade_Rounded"]
    .map(grade_names)
)

comparison["Absolute_Error"] = (
    abs(
        comparison["Actual_Grade"]
        - comparison["Predicted_Grade"]
    )
)

comparison.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ==========================================
# Actual vs Predicted Plot
# ==========================================

print("\nCreating Actual vs Predicted plot...")

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.plot(
    [0, 7],
    [0, 7],
    linestyle="--"
)

plt.xlabel(
    "Actual Grade"
)

plt.ylabel(
    "Predicted Grade"
)

plt.title(
    "Actual vs Predicted Grades"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR
    / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Residual Analysis
# ==========================================

print(
    "\nCreating residual analysis..."
)

residuals = (
    y_test.values
    - predictions
)

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    residuals,
    bins=15,
    kde=True
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Residual Distribution"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR
    / "residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Residual Scatter Plot
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Grade"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residuals vs Predicted Values"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR
    / "residuals_vs_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# Feature Importance
# ==========================================

print(
    "\nAnalyzing feature importance..."
)

estimator = model.named_steps["model"]

preprocessor = (
    model.named_steps["preprocessor"]
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ==========================================
# Tree-Based Models
# ==========================================

if hasattr(
    estimator,
    "feature_importances_"
):

    importance = (
        estimator
        .feature_importances_
    )

    importance_type = (
        "Feature Importance"
    )


# ==========================================
# Linear Models
# ==========================================

elif hasattr(
    estimator,
    "coef_"
):

    importance = np.abs(
        estimator.coef_
    )

    importance_type = (
        "Absolute Coefficient"
    )


# ==========================================
# Unknown Model
# ==========================================

else:

    importance = None

    importance_type = None


if importance is not None:

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    print(
        "\nTop 10 Important Features:"
    )

    print(
        importance_df.head(10)
        .to_string(index=False)
    )

    # --------------------------------------
    # Feature Importance Plot
    # --------------------------------------

    top10 = (
        importance_df
        .head(10)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top10["Feature"],
        top10["Importance"]
    )

    plt.xlabel(
        importance_type
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 10 Most Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ==========================================
# Save Evaluation Metrics
# ==========================================

metrics = {
    "model": type(
        estimator
    ).__name__,
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "r2": float(r2),
    "test_samples": int(len(y_test))
}

with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================
# Final Summary
# ==========================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Model : "
    f"{type(estimator).__name__}"
)

print(
    f"MAE   : {mae:.4f}"
)

print(
    f"RMSE  : {rmse:.4f}"
)

print(
    f"R²    : {r2:.4f}"
)

print("\nSaved evaluation files:")

print(
    f"Metrics      : {METRICS_PATH}"
)

print(
    f"Predictions  : {PREDICTIONS_PATH}"
)

if importance is not None:

    print(
        f"Importance   : {IMPORTANCE_PATH}"
    )

print(
    f"Plots        : {REPORTS_DIR}"
)

print("=" * 70)