# ==========================================
# File: src/predict.py
# Purpose: Predict student performance
# ==========================================

from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.pkl"
)


# ==========================================
# Grade Mapping
# ==========================================

GRADE_MAPPING = {
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
# Load Model
# ==========================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Model not found.

Expected path:
{MODEL_PATH}

Run training first:

python -m src.train
"""
        )

    model = joblib.load(MODEL_PATH)

    return model


# ==========================================
# Predict Student Grade
# ==========================================

def predict_student(student_data: dict):

    # --------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------

    student_df = pd.DataFrame(
        [student_data]
    )

    # --------------------------------------
    # Load trained pipeline
    # --------------------------------------

    model = load_model()

    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(
        student_df
    )

    predicted_grade = round(
        float(prediction[0])
    )

    # --------------------------------------
    # Keep prediction inside valid range
    # --------------------------------------

    predicted_grade = max(
        0,
        min(7, predicted_grade)
    )

    grade_label = GRADE_MAPPING[
        predicted_grade
    ]

    return predicted_grade, grade_label


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTOR")
    print("=" * 60)

    # --------------------------------------
    # Example Student
    # --------------------------------------

    student = {

        "Student_Age": 18.0,

        "Sex": "Male",

        "High_School_Type": "State",

        "Scholarship": 50.0,

        "Additional_Work": 0.0,

        "Sports_activity": 0.0,

        "Transportation": "Private",

        "Weekly_Study_Hours": 8,

        "Attendance": 2.0,

        "Reading": 1.0,

        "Notes": 1.0,

        "Listening_in_Class": 1.0,

        "Project_work": 1.0
    }

    # --------------------------------------
    # Predict
    # --------------------------------------

    predicted_grade, grade_label = predict_student(
        student
    )

    # --------------------------------------
    # Display
    # --------------------------------------

    print("\nStudent Information:")
    print("-" * 60)

    for feature, value in student.items():

        print(
            f"{feature:<25}: {value}"
        )

    print("\nPrediction:")
    print("-" * 60)

    print(
        f"Predicted Grade Score : "
        f"{predicted_grade}"
    )

    print(
        f"Predicted Grade       : "
        f"{grade_label}"
    )

    print("=" * 60)