# ======================================================
# File: app.py
# Student Performance Predictor + SHAP Explainability
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ======================================================
# LOAD TRAINED MODEL
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

pipeline = joblib.load(MODEL_PATH)

# Extract pipeline components
preprocessor = pipeline.named_steps["preprocessor"]
rf_model = pipeline.named_steps["model"]

# ======================================================
# LABELS
# ======================================================

GRADE_LABELS = {
    0: "Fail",
    1: "DD",
    2: "DC",
    3: "CC",
    4: "CB",
    5: "BB",
    6: "BA",
    7: "AA",
}

# ======================================================
# NUMERIC MAPPINGS
# Must match data_cleaning.py
# ======================================================

AGE_MAP = {
    "18": 18.0,
    "19-22": 20.5,
    "23-27": 25.0
}

SCHOLARSHIP_MAP = {
    "25%": 25.0,
    "50%": 50.0,
    "75%": 75.0,
    "100%": 100.0
}

ATTENDANCE_MAP = {
    "Never": 0.0,
    "Sometimes": 1.0,
    "Always": 2.0
}

# ======================================================
# HEADER
# ======================================================

st.title("🎓 Student Performance Predictor")

st.markdown(
    """
Predict the **final academic grade** using a trained **Random Forest Regression Model**.

The model uses student demographic information, study habits, attendance, scholarship and classroom behavior.
"""
)

st.divider()

# ======================================================
# INPUT SECTION
# ======================================================

left, right = st.columns(2)

with left:

    age = st.selectbox(
        "Student Age",
        ["18", "19-22", "23-27"]
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    high_school = st.selectbox(
        "High School Type",
        ["State", "Private", "Other"]
    )

    scholarship = st.selectbox(
        "Scholarship",
        ["25%", "50%", "75%", "100%"]
    )

    transportation = st.selectbox(
        "Transportation",
        ["Private", "Bus"]
    )

with right:

    study_hours = st.selectbox(
        "Weekly Study Hours",
        [0, 2, 8, 12]
    )

    attendance = st.selectbox(
        "Attendance",
        ["Never", "Sometimes", "Always"]
    )

    additional_work = st.radio(
        "Additional Work",
        ["No", "Yes"]
    )

    sports = st.radio(
        "Sports Activity",
        ["No", "Yes"]
    )

st.subheader("Study Habits")

c1, c2, c3, c4 = st.columns(4)

with c1:
    reading = st.checkbox("Reading")

with c2:
    notes = st.checkbox("Taking Notes")

with c3:
    listening = st.checkbox("Listening in Class")

with c4:
    project = st.checkbox("Project Work")

st.divider()

# ======================================================
# PREDICTION
# ======================================================

if st.button("🚀 Predict Grade", use_container_width=True):

    # -----------------------------
    # Convert UI to training format
    # -----------------------------

    input_df = pd.DataFrame([{
        "Student_Age": AGE_MAP[age],
        "Sex": sex,
        "High_School_Type": high_school,
        "Scholarship": SCHOLARSHIP_MAP[scholarship],
        "Additional_Work": 1.0 if additional_work == "Yes" else 0.0,
        "Sports_activity": 1.0 if sports == "Yes" else 0.0,
        "Transportation": transportation,
        "Weekly_Study_Hours": float(study_hours),
        "Attendance": ATTENDANCE_MAP[attendance],
        "Reading": 1.0 if reading else 0.0,
        "Notes": 1.0 if notes else 0.0,
        "Listening_in_Class": 1.0 if listening else 0.0,
        "Project_work": 1.0 if project else 0.0,
    }])

    # -----------------------------
    # Predict
    # -----------------------------

    prediction = pipeline.predict(input_df)[0]

    grade_score = int(round(prediction))
    grade_score = max(0, min(7, grade_score))

    grade = GRADE_LABELS[grade_score]

    st.success("Prediction completed successfully!")

    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            "Predicted Grade",
            grade
        )

    with m2:
        st.metric(
            "Grade Score",
            grade_score
        )

    st.divider()

    # ======================================================
    # SHAP EXPLANATION
    # ======================================================

    st.subheader("🧠 Why did the model predict this grade?")

    X_processed = preprocessor.transform(input_df)

    # Convert sparse -> dense if needed
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_processed)

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

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    st.pyplot(fig)

    plt.close(fig)

    # ======================================================
    # STUDENT DATA
    # ======================================================

    st.divider()

    st.subheader("📋 Processed Student Data")

    st.dataframe(
        input_df,
        use_container_width=True
    )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Machine Learning Portfolio Project • Random Forest • SHAP • Streamlit"
)