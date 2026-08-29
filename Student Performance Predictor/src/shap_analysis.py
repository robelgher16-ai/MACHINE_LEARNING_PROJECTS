# ==========================================
# File: src/shap_analysis.py
# Purpose: Explain Random Forest predictions
# using SHAP (Explainable AI)
# ==========================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.data_cleaning import clean_dataset

warnings.filterwarnings("ignore")


# ==========================================
# Project Paths
# ==========================================

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

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# Header
# ==========================================

print("=" * 70)
print("STUDENT PERFORMANCE SHAP ANALYSIS")
print("=" * 70)


# ==========================================
# 1. Load Dataset
# ==========================================

print("\n[1/7] Loading dataset...")

df = pd.read_csv(DATA_PATH)
df = clean_dataset(df)

X = df.drop(
    columns=[
        "Grade",
        "Student_ID"
    ]
)

print(f"Samples : {len(X)}")
print(f"Features: {X.shape[1]}")


# ==========================================
# 2. Load Trained Pipeline
# ==========================================

print("\n[2/7] Loading trained model...")

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

print("Model loaded successfully.")
print(f"Model type: {type(model).__name__}")


# ==========================================
# 3. Transform Features
# ==========================================

print("\n[3/7] Transforming features...")

X_processed = preprocessor.transform(X)

feature_names = (
    preprocessor.get_feature_names_out()
)

print(f"Processed shape: {X_processed.shape}")


# ==========================================
# 4. Compute SHAP Values
# ==========================================

print("\n[4/7] Computing SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_processed)

# Regression output safety
if isinstance(shap_values, list):
    shap_values = shap_values[0]

print("SHAP values computed.")


# ==========================================
# 5. SHAP Summary Plot
# ==========================================

print("\n[5/7] Creating SHAP summary plot...")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    show=False
)

summary_path = (
    REPORTS_DIR
    / "shap_summary.png"
)

plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()

print("Summary plot saved.")


# ==========================================
# 6. SHAP Bar Plot
# ==========================================

print("\n[6/7] Creating SHAP bar plot...")

importance = np.abs(shap_values).mean(axis=0)

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
)

plt.figure(figsize=(10, 6))

top = importance_df.head(15)

plt.barh(
    top["Feature"][::-1],
    top["Importance"][::-1]
)

plt.xlabel("Mean |SHAP Value|")
plt.title("Top 15 Important Features")
plt.tight_layout()

bar_path = (
    REPORTS_DIR
    / "shap_bar.png"
)

plt.savefig(bar_path, dpi=300)
plt.close()

importance_csv = (
    REPORTS_DIR
    / "shap_importance.csv"
)

importance_df.to_csv(
    importance_csv,
    index=False
)

print("Bar plot saved.")
print("Importance table saved.")


# ==========================================
# 7. Waterfall Plot (Student #1)
# ==========================================

print("\n[7/7] Creating waterfall explanation...")

waterfall_path = (
    REPORTS_DIR / "shap_waterfall.png"
)

# SHAP expected value may be an array -> convert to scalar
base_value = explainer.expected_value

if isinstance(base_value, np.ndarray):
    base_value = float(base_value.flatten()[0])
else:
    base_value = float(base_value)

explanation = shap.Explanation(
    values=shap_values[0],
    base_values=base_value,
    data=X_processed[0],
    feature_names=feature_names
)

plt.figure(figsize=(10, 7))

shap.plots.waterfall(
    explanation,
    max_display=12,
    show=False
)

plt.tight_layout()
plt.savefig(waterfall_path, dpi=300)
plt.close()

print("Waterfall plot saved.")
# ==========================================
# Save SHAP Values
# ==========================================

shap_csv = (
    REPORTS_DIR
    / "shap_values.csv"
)

pd.DataFrame(
    shap_values,
    columns=feature_names
).to_csv(
    shap_csv,
    index=False
)


# ==========================================
# Final Report
# ==========================================

print("\n")
print("=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(f"Summary Plot   : {summary_path}")
print(f"Bar Plot       : {bar_path}")
print(f"Waterfall Plot : {waterfall_path}")
print(f"Importance CSV : {importance_csv}")
print(f"SHAP Values    : {shap_csv}")

print("=" * 70)