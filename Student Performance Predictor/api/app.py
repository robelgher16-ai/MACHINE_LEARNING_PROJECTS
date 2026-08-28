from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("models/best_model.pkl")

app = FastAPI(
    title="Student Performance Prediction API",
    version="1.0"
)

# -------------------------
# Input Schema
# -------------------------
class Student(BaseModel):
    Student_ID: str
    Student_Age: str
    Sex: str
    High_School_Type: str
    Scholarship: str
    Additional_Work: str
    Sports_activity: str
    Transportation: str
    Weekly_Study_Hours: int
    Attendance: str
    Reading: str
    Notes: str
    Listening_in_Class: str
    Project_work: str

# Grade decoding
grade_decode = {
    0:"FF",
    1:"DD",
    2:"DC",
    3:"CC",
    4:"CB",
    5:"BB",
    6:"BA",
    7:"AA"
}

# -------------------------
# Home
# -------------------------
@app.get("/")
def home():
    return {
        "message":"Student Performance Prediction API"
    }

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status":"running"}

# -------------------------
# Predict
# -------------------------
@app.post("/predict")
def predict(student: Student):

    data = pd.DataFrame([student.dict()])

    prediction = model.predict(data)[0]

    prediction = int(round(prediction))

    prediction = max(0, min(7, prediction))

    return {
        "Predicted_Grade": grade_decode[prediction]
    }


from typing import List

@app.post("/batch_predict")
def batch_predict(students: List[Student]):

    df = pd.DataFrame([s.dict() for s in students])

    preds = model.predict(df)

    results = []

    for p in preds:

        p = int(round(p))
        p = max(0, min(7, p))

        results.append(grade_decode[p])

    return {
        "Predictions": results
    }