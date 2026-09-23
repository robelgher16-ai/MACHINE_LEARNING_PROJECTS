import streamlit as st
import pandas as pd
import requests

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ======================================================
# API URL (LOCAL)
# ======================================================

API_URL = "https://student-performance-api-3rxe.onrender.com/predict"

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>
.stApp{
    background: linear-gradient(135deg,#eef5ff,#f8faff);
}

.hero{
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding:30px;
    border-radius:20px;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.hero h1{
    font-size:40px;
    margin:0;
}

.card{
    background:white;
    padding:18px;
    border-radius:16px;
    box-shadow:0 8px 18px rgba(0,0,0,.08);
    border-left:5px solid #2563eb;
}

.result{
    background:linear-gradient(135deg,#16a34a,#059669);
    color:white;
    padding:22px;
    border-radius:16px;
    text-align:center;
}

section[data-testid="stSidebar"]{
    background:#172554;
}

section[data-testid="stSidebar"] *{
    color:white;
}

[data-testid="stFileUploader"]{
    background:white;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LABELS
# ======================================================

AGE_MAP = {
    "18":18.0,
    "19-22":20.5,
    "23-27":25.0
}

SCHOLARSHIP_MAP = {
    "25%":25.0,
    "50%":50.0,
    "75%":75.0,
    "100%":100.0
}

ATTENDANCE_MAP = {
    "Never":0.0,
    "Sometimes":1.0,
    "Always":2.0
}

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🎓 Student Predictor")

st.sidebar.markdown("### Model")
st.sidebar.write("Random Forest")

st.sidebar.markdown("### Features")
st.sidebar.write("13 Student Features")

st.sidebar.markdown("### Grades")
st.sidebar.write("Fail → AA")

st.sidebar.markdown("---")
st.sidebar.success("FastAPI Connected")

# ======================================================
# HERO
# ======================================================

st.markdown("""
<div class="hero">
<h1>Student Performance Predictor</h1>
<p>Machine Learning Academic Grade Prediction using Random Forest + FastAPI + Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INFO CARDS
# ======================================================

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
    <h3>Model</h3>
    <h2>Random Forest</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h3>Features</h3>
    <h2>13</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h3>Output</h3>
    <h2>8 Grades</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ======================================================
# INPUT FORM
# ======================================================

left,right = st.columns(2)

with left:

    age = st.selectbox("Student Age", ["18","19-22","23-27"])

    sex = st.selectbox("Sex", ["Male","Female"])

    school = st.selectbox(
        "High School Type",
        ["State","Private","Other"]
    )

    scholarship = st.selectbox(
        "Scholarship",
        ["25%","50%","75%","100%"]
    )

    transport = st.selectbox(
        "Transportation",
        ["Private","Bus"]
    )

with right:

    study = st.select_slider(
        "Weekly Study Hours",
        options=[0,2,8,12]
    )

    attendance = st.selectbox(
        "Attendance",
        ["Never","Sometimes","Always"]
    )

    work = st.radio(
        "Additional Work",
        ["No","Yes"]
    )

    sports = st.radio(
        "Sports Activity",
        ["No","Yes"]
    )

st.markdown("### 📚 Study Habits")

s1,s2,s3,s4 = st.columns(4)

with s1:
    reading = st.checkbox("Reading")

with s2:
    notes = st.checkbox("Taking Notes")

with s3:
    listening = st.checkbox("Listening")

with s4:
    project = st.checkbox("Project Work")

st.divider()

# ======================================================
# PREDICT BUTTON
# ======================================================

if st.button("🚀 Predict Academic Grade", use_container_width=True):

    df = pd.DataFrame([{
        "Student_Age": AGE_MAP[age],
        "Sex": sex,
        "High_School_Type": school,
        "Scholarship": SCHOLARSHIP_MAP[scholarship],
        "Additional_Work": 1.0 if work=="Yes" else 0.0,
        "Sports_activity": 1.0 if sports=="Yes" else 0.0,
        "Transportation": transport,
        "Weekly_Study_Hours": float(study),
        "Attendance": ATTENDANCE_MAP[attendance],
        "Reading": 1.0 if reading else 0.0,
        "Notes": 1.0 if notes else 0.0,
        "Listening_in_Class": 1.0 if listening else 0.0,
        "Project_work": 1.0 if project else 0.0
    }])

    try:

        response = requests.post(
            API_URL,
            json=df.iloc[0].to_dict(),
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        st.markdown(f"""
        <div class="result">
            <p>Predicted Academic Grade</p>
            <h1>{result['predicted_grade']}</h1>
            <h3>Score: {result['predicted_score']}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.subheader("Student Information")
        st.dataframe(df, use_container_width=True)

    except requests.exceptions.RequestException:
        st.error("Could not connect to the FastAPI server.")

# ======================================================
# SAMPLE STUDENT PROFILES
# ======================================================

with st.expander("🎯 Hidden Sample Student Profiles"):

    st.markdown("Use these profiles to quickly test the model.")

    st.code("""\nHigh Performer\nAge: 19-22\nStudy Hours: 12\nAttendance: Always\nReading: Yes\nNotes: Yes\nListening: Yes\nProject: Yes\nScholarship: 100%\n""")

    st.code("""\nAverage Student\nAge: 19-22\nStudy Hours: 8\nAttendance: Sometimes\nReading: Yes\nNotes: Yes\nListening: No\nProject: Yes\nScholarship: 50%\n""")

    st.code("""\nAt-Risk Student\nAge: 18\nStudy Hours: 0\nAttendance: Never\nReading: No\nNotes: No\nListening: No\nProject: No\nScholarship: 25%\n""")

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Machine Learning Portfolio Project • Random Forest • FastAPI • Streamlit"
)