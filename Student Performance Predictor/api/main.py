
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Student Performance Predictor API",
    description=(
        "REST API for predicting student academic performance "
        "using a trained machine learning model."
    ),
    version="1.0.0",
)


# ============================================================
# INPUT DATA SCHEMA
# ============================================================

class StudentInput(BaseModel):
    Student_Age: float
    Sex: str
    High_School_Type: str
    Scholarship: float
    Additional_Work: float
    Sports_activity: float
    Transportation: str
    Weekly_Study_Hours: float
    Attendance: float
    Reading: float
    Notes: float
    Listening_in_Class: float
    Project_work: float


# ============================================================
# GRADE MAPPING
# ============================================================

GRADE_MAPPING = {
    0: "Fail",
    1: "DD",
    2: "DC",
    3: "CC",
    4: "CB",
    5: "BB",
    6: "BA",
    7: "AA",
}


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Student Performance Predictor API is running",
        "status": "healthy",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    if not MODEL_LOADED:
        return {
            "status": "unhealthy",
            "model": "not loaded",
            "error": MODEL_LOAD_ERROR,
        }

    return {
        "status": "healthy",
        "model": "loaded",
        "model_path": str(MODEL_PATH),
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(student: StudentInput):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_LOADED or model is None:
        raise HTTPException(
            status_code=500,
            detail="Trained model could not be loaded.",
        )

    # --------------------------------------------------------
    # Convert Pydantic input to dictionary
    # --------------------------------------------------------

    input_data = {
        "Student_Age": student.Student_Age,
        "Sex": student.Sex,
        "High_School_Type": student.High_School_Type,
        "Scholarship": student.Scholarship,
        "Additional_Work": student.Additional_Work,
        "Sports_activity": student.Sports_activity,
        "Transportation": student.Transportation,
        "Weekly_Study_Hours": student.Weekly_Study_Hours,
        "Attendance": student.Attendance,
        "Reading": student.Reading,
        "Notes": student.Notes,
        "Listening_in_Class": student.Listening_in_Class,
        "Project_work": student.Project_work,
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # The trained sklearn pipeline expects a DataFrame
    # with the original feature names.
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    # --------------------------------------------------------
    # Convert prediction to integer grade score
    # --------------------------------------------------------

    predicted_score = int(round(float(prediction)))

    # Keep score inside the expected grade range
    predicted_score = max(0, min(7, predicted_score))

    # --------------------------------------------------------
    # Convert numerical score to grade
    # --------------------------------------------------------

    predicted_grade = GRADE_MAPPING.get(
        predicted_score,
        "Unknown",
    )

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return {
        "predicted_score": predicted_score,
        "predicted_grade": predicted_grade,
    }

