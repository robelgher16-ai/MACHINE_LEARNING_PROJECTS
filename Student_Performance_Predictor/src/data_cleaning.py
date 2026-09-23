
# ==========================================
# File: src/data_cleaning.py
# Purpose: Clean and validate student data
# ==========================================

import numpy as np
import pandas as pd


# ==========================================
# Expected Columns
# ==========================================

EXPECTED_COLUMNS = [
    "Student_ID",
    "Student_Age",
    "Sex",
    "High_School_Type",
    "Scholarship",
    "Additional_Work",
    "Sports_activity",
    "Transportation",
    "Weekly_Study_Hours",
    "Attendance",
    "Reading",
    "Notes",
    "Listening_in_Class",
    "Project_work",
    "Grade",
]


# ==========================================
# Grade Mapping
# ==========================================

GRADE_MAPPING = {
    "Fail": 0,
    "FF": 0,
    "DD": 1,
    "DC": 2,
    "CC": 3,
    "CB": 4,
    "BB": 5,
    "BA": 6,
    "AA": 7,
}


# ==========================================
# Clean Dataset
# ==========================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the Student Performance dataset.

    Cleaning steps:
        1. Validate expected columns
        2. Remove duplicate rows
        3. Clean text values
        4. Convert invalid values to NaN
        5. Convert Student_Age to numeric
        6. Convert Scholarship to numeric
        7. Convert Yes/No features to numeric
        8. Convert Attendance to ordinal numeric
        9. Convert Weekly_Study_Hours to numeric
        10. Convert Grade to numeric
        11. Convert pandas NA to numpy NaN
        12. Validate target

    Missing feature values are NOT removed.
    They are handled later by the preprocessing pipeline.
    """

    # Make a copy so the original dataframe is not modified
    df = df.copy()

    # ==========================================
    # 1. Validate Columns
    # ==========================================

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # ==========================================
    # 2. Remove Duplicate Rows
    # ==========================================

    before_duplicates = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    removed_duplicates = (
        before_duplicates - len(df)
    )

    # ==========================================
    # 3. Clean Text Columns
    # ==========================================

    text_columns = [
        "Sex",
        "High_School_Type",
        "Scholarship",
        "Additional_Work",
        "Sports_activity",
        "Transportation",
        "Attendance",
        "Reading",
        "Notes",
        "Listening_in_Class",
        "Project_work",
        "Grade",
        "Student_Age",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        # Convert empty strings to missing values
        df[column] = df[column].replace(
            "",
            np.nan
        )

    # ==========================================
    # 4. Convert Suspicious Values to Missing
    # ==========================================

    # ------------------------------------------
    # Attendance
    # Expected:
    # Always / Sometimes / Never
    #
    # "3" is an invalid value in the dataset.
    # ------------------------------------------

    df["Attendance"] = df["Attendance"].replace(
        {
            "3": np.nan,
            3: np.nan,
        }
    )

    # ------------------------------------------
    # Notes
    # Expected:
    # Yes / No
    #
    # "6" is an invalid value in the dataset.
    # ------------------------------------------

    df["Notes"] = df["Notes"].replace(
        {
            "6": np.nan,
            6: np.nan,
        }
    )

    # ------------------------------------------
    # Listening_in_Class
    # Expected:
    # Yes / No
    #
    # "6" is an invalid value in the dataset.
    # ------------------------------------------

    df["Listening_in_Class"] = (
        df["Listening_in_Class"].replace(
            {
                "6": np.nan,
                6: np.nan,
            }
        )
    )

    # ==========================================
    # 5. Convert Student Age
    # ==========================================

    age_mapping = {
        "18": 18.0,
        "19-22": 20.5,
        "23-27": 25.0,
    }

    df["Student_Age"] = (
        df["Student_Age"]
        .map(age_mapping)
        .astype("float64")
    )

    # ==========================================
    # 6. Convert Scholarship
    # ==========================================

    scholarship_mapping = {
        "25%": 25.0,
        "50%": 50.0,
        "75%": 75.0,
        "100%": 100.0,
    }

    df["Scholarship"] = (
        df["Scholarship"]
        .map(scholarship_mapping)
        .astype("float64")
    )

    # ==========================================
    # 7. Convert Yes/No Features to Binary
    # ==========================================

    binary_columns = [
        "Additional_Work",
        "Sports_activity",
        "Reading",
        "Notes",
        "Listening_in_Class",
        "Project_work",
    ]

    binary_mapping = {
        "No": 0.0,
        "Yes": 1.0,
    }

    for column in binary_columns:

        df[column] = (
            df[column]
            .map(binary_mapping)
            .astype("float64")
        )

    # ==========================================
    # 8. Convert Attendance to Ordinal Numeric
    # ==========================================

    attendance_mapping = {
        "Never": 0.0,
        "Sometimes": 1.0,
        "Always": 2.0,
    }

    df["Attendance"] = (
        df["Attendance"]
        .map(attendance_mapping)
        .astype("float64")
    )

    # ==========================================
    # 9. Convert Weekly Study Hours
    # ==========================================

    df["Weekly_Study_Hours"] = pd.to_numeric(
        df["Weekly_Study_Hours"],
        errors="coerce"
    )

    # ==========================================
    # 10. Convert Grade
    # ==========================================

    original_grades = df["Grade"].copy()

    df["Grade"] = (
        df["Grade"]
        .map(GRADE_MAPPING)
    )

    # ==========================================
    # 11. Validate Grade
    # ==========================================

    if df["Grade"].isna().any():

        invalid_grades = (
            original_grades[
                df["Grade"].isna()
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unknown or missing Grade values detected.\n"
            f"Invalid values: {invalid_grades}\n"
            f"Expected values: {list(GRADE_MAPPING.keys())}"
        )

    # Convert target to integer
    df["Grade"] = df["Grade"].astype("int64")

    # ==========================================
    # 12. Convert pandas NA to numpy NaN
    # ==========================================

    # This is important for compatibility
    # with scikit-learn SimpleImputer.
    df = df.where(
        pd.notna(df),
        np.nan
    )

    # ==========================================
    # 13. Validation Summary
    # ==========================================

    print(
        f"Removed duplicate rows: "
        f"{removed_duplicates}"
    )

    print(
        "Cleaning completed successfully."
    )

    return df


# ==========================================
# Prepare Features and Target
# ==========================================

def prepare_features(
    df: pd.DataFrame,
):
    """
    Clean dataset and separate features and target.

    Student_ID is excluded because it is an
    identifier and should not be used as a
    predictive feature.
    """

    # Clean dataset
    df = clean_dataset(df)

    # ------------------------------------------
    # Features
    # ------------------------------------------

    X = df.drop(
        columns=[
            "Grade",
            "Student_ID",
        ]
    )

    # ------------------------------------------
    # Target
    # ------------------------------------------

    y = df["Grade"].astype("int64")

    return X, y
