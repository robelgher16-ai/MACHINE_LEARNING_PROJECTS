# ==========================================
# File: src/predict.py
# Purpose: Predict student grades
# ==========================================

import joblib
import pandas as pd

# Load trained model once
model = joblib.load("models/best_model.pkl")

# Convert numeric prediction back to grade
grade_decode = {
    0: "FF",
    1: "DD",
    2: "DC",
    3: "CC",
    4: "CB",
    5: "BB",
    6: "BA",
    7: "AA"
}


def predict_single(student_data: dict):
    """
    Predict one student's grade.

    Parameters
    ----------
    student_data : dict

    Returns
    -------
    str
        Predicted letter grade.
    """

    df = pd.DataFrame([student_data])

    prediction = model.predict(df)[0]

    prediction = int(round(prediction))

    prediction = max(0, min(7, prediction))

    return grade_decode[prediction]


def predict_batch(students: list):
    """
    Predict multiple students.

    Parameters
    ----------
    students : list of dictionaries

    Returns
    -------
    list
        Predicted grades.
    """

    df = pd.DataFrame(students)

    predictions = model.predict(df)

    results = []

    for p in predictions:

        p = int(round(p))

        p = max(0, min(7, p))

        results.append(grade_decode[p])

    return results