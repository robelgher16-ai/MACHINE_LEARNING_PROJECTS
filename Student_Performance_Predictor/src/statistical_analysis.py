# ==========================================
# File: src/statistical_analysis.py
# Purpose: Statistical analysis of features
#          against student Grade
# ==========================================

from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import (
    pearsonr,
    spearmanr,
    pointbiserialr,
    f_oneway
)

from src.data_cleaning import clean_dataset


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


# ==========================================
# Load Dataset
# ==========================================

print("=" * 75)
print("STUDENT PERFORMANCE STATISTICAL ANALYSIS")
print("=" * 75)


df_raw = pd.read_csv(
    DATA_PATH
)


df = clean_dataset(
    df_raw
)


# ==========================================
# Numerical Features
# ==========================================

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


# ==========================================
# Spearman Correlation
# ==========================================

print("\n[1/3] Spearman correlation")


spearman_results = []


for feature in numerical_features:

    valid = df[
        [feature, "Grade"]
    ].dropna()


    correlation, p_value = spearmanr(
        valid[feature],
        valid["Grade"]
    )


    spearman_results.append({
        "Feature": feature,
        "Spearman_Rho": correlation,
        "P_Value": p_value
    })


spearman_df = pd.DataFrame(
    spearman_results
)


spearman_df = (
    spearman_df
    .sort_values(
        "Spearman_Rho",
        ascending=False
    )
)


print(
    spearman_df.to_string(
        index=False
    )
)


# ==========================================
# Pearson Correlation
# ==========================================

print("\n[2/3] Pearson correlation")


pearson_results = []


for feature in numerical_features:

    valid = df[
        [feature, "Grade"]
    ].dropna()


    correlation, p_value = pearsonr(
        valid[feature],
        valid["Grade"]
    )


    pearson_results.append({
        "Feature": feature,
        "Pearson_R": correlation,
        "P_Value": p_value
    })


pearson_df = pd.DataFrame(
    pearson_results
)


pearson_df = (
    pearson_df
    .sort_values(
        "Pearson_R",
        ascending=False
    )
)


print(
    pearson_df.to_string(
        index=False
    )
)


# ==========================================
# Categorical Feature Analysis
# ==========================================

print("\n[3/3] Categorical feature analysis")


categorical_features = [
    "Sex",
    "High_School_Type",
    "Transportation",
]


for feature in categorical_features:

    print("\n" + "=" * 60)

    print(
        f"{feature} vs Grade"
    )

    print("=" * 60)


    groups = []


    for category in df[feature].dropna().unique():

        values = (
            df.loc[
                df[feature] == category,
                "Grade"
            ]
            .dropna()
        )


        if len(values) > 1:
            groups.append(values)


    if len(groups) >= 2:

        statistic, p_value = f_oneway(
            *groups
        )


        print(
            f"ANOVA F-statistic: "
            f"{statistic:.4f}"
        )

        print(
            f"ANOVA p-value: "
            f"{p_value:.4f}"
        )


    else:

        print(
            "Not enough groups for ANOVA."
        )


# ==========================================
# Final
# ==========================================

print("\n")


print("=" * 75)

print(
    "STATISTICAL ANALYSIS COMPLETE"
)

print("=" * 75)