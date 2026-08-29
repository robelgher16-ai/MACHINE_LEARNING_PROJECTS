# ==========================================
# File: src/analyze_data.py
# Purpose: Exploratory analysis for the
#          Student Performance Predictor
# ==========================================

from pathlib import Path

import pandas as pd
import numpy as np

from src.data_cleaning import clean_dataset


# ==========================================
# Configuration
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


# ==========================================
# Load Dataset
# ==========================================

print("=" * 75)
print("STUDENT PERFORMANCE DATA ANALYSIS")
print("=" * 75)


print("\n[1/8] Loading dataset...")


df_raw = pd.read_csv(
    DATA_PATH
)


print(
    f"Dataset shape: {df_raw.shape}"
)


# ==========================================
# Clean Dataset
# ==========================================

print("\n[2/8] Cleaning dataset...")


df = clean_dataset(
    df_raw
)


print(
    "Cleaning completed."
)


# ==========================================
# Basic Information
# ==========================================

print("\n[3/8] Dataset information")


print("\nData types:")
print(
    df.dtypes
)


print("\nMissing values:")

print(
    df.isna().sum()
)


# ==========================================
# Target Distribution
# ==========================================

print("\n[4/8] Grade distribution")


grade_counts = (
    df["Grade"]
    .value_counts()
    .sort_index()
)


print(
    grade_counts
)


print("\nGrade percentages:")


grade_percentages = (
    df["Grade"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


print(
    grade_percentages.round(2)
)


# ==========================================
# Numerical Feature Statistics
# ==========================================

print(
    "\n[5/8] Numerical feature statistics"
)


numerical_features = [
    "Student_Age",
    "Scholarship",
    "Additional_Work",
    "Sports_activity",
    "Weekly_Study_Hours",
    "Attendance",
    "Reading",
    "Notes",
    "Listening_in_Class",
    "Project_work",
]


print(
    df[numerical_features]
    .describe()
    .T
)


# ==========================================
# Feature Means by Grade
# ==========================================

print(
    "\n[6/8] Feature averages by Grade"
)


grade_means = (
    df.groupby("Grade")[
        numerical_features
    ]
    .mean()
    .round(3)
)


print(
    grade_means
)


# ==========================================
# Correlation with Grade
# ==========================================

print(
    "\n[7/8] Numerical correlation with Grade"
)


correlation = (
    df[
        numerical_features + ["Grade"]
    ]
    .corr()["Grade"]
    .drop("Grade")
    .sort_values(
        ascending=False
    )
)


print(
    correlation.round(4)
)


# ==========================================
# Categorical Feature Analysis
# ==========================================

print(
    "\n[8/8] Categorical feature analysis"
)


categorical_features = [
    "Sex",
    "High_School_Type",
    "Transportation",
]


for feature in categorical_features:

    print("\n" + "-" * 60)

    print(
        f"{feature} vs Grade"
    )

    print("-" * 60)


    grouped = (
        df.groupby(feature)["Grade"]
        .agg(
            [
                "count",
                "mean",
                "std"
            ]
        )
        .round(3)
    )


    print(
        grouped
    )


# ==========================================
# Final
# ==========================================

print("\n")


print("=" * 75)

print(
    "DATA ANALYSIS COMPLETE"
)

print("=" * 75)