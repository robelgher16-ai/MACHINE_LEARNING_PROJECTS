# ==========================================
# File: src/evaluate.py
# Purpose: Evaluate the trained model
# ==========================================

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("data/raw/student_performance.csv")

grade_mapping = {
    "FF": 0,
    "DD": 1,
    "DC": 2,
    "CC": 3,
    "CB": 4,
    "BB": 5,
    "BA": 6,
    "AA": 7
}

df["Grade"] = df["Grade"].map(grade_mapping)

X = df.drop("Grade", axis=1)
y = df["Grade"]


# ------------------------------------------
# Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------
# Load Best Model
# ------------------------------------------

model = joblib.load("models/best_model.pkl")


# ------------------------------------------
# Prediction
# ------------------------------------------

predictions = model.predict(X_test)


# ------------------------------------------
# Metrics
# ------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

metrics = {
    "MAE": round(float(mae), 3),
    "RMSE": round(float(rmse), 3),
    "R2": round(float(r2), 3)
}


# ------------------------------------------
# Save Metrics
# ------------------------------------------

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")
print("=" * 40)