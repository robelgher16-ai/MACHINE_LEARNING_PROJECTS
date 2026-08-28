import streamlit as st
import requests

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 35px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.prediction-grade {
    font-size: 42px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">🎓 Student Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning System for Predicting Student Academic Performance'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# STUDENT INFORMATION
# ==========================================

st.markdown(
    '<div class="section-title">👤 Student Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    student_id = st.text_input(
        "Student ID",
        "ST001"
    )

    student_age = st.selectbox(
        "Student Age",
        ["18-21", "22-25", "26-30", "31-35"]
    )

with col2:

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    high_school_type = st.selectbox(
        "High School Type",
        ["Private", "Public", "Other"]
    )

with col3:

    scholarship = st.selectbox(
        "Scholarship",
        ["None", "25%", "50%", "75%", "100%"]
    )

    transportation = st.selectbox(
        "Transportation",
        ["Bus", "Private", "Other"]
    )


# ==========================================
# ACADEMIC INFORMATION
# ==========================================

st.markdown(
    '<div class="section-title">📚 Academic Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    weekly_study_hours = st.number_input(
        "Weekly Study Hours",
        min_value=0,
        max_value=100,
        value=15
    )

    attendance = st.selectbox(
        "Attendance",
        ["Always", "Sometimes", "Never"]
    )

with col2:

    reading = st.selectbox(
        "Reading",
        ["Yes", "No"]
    )

    notes = st.selectbox(
        "Taking Notes",
        ["Yes", "No"]
    )

with col3:

    listening = st.selectbox(
        "Listening in Class",
        ["Yes", "No"]
    )

    project_work = st.selectbox(
        "Project Work",
        ["Yes", "No"]
    )


# ==========================================
# LIFESTYLE INFORMATION
# ==========================================

st.markdown(
    '<div class="section-title">⚽ Lifestyle & Activities</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    additional_work = st.selectbox(
        "Additional Work",
        ["Yes", "No"]
    )

with col2:

    sports_activity = st.selectbox(
        "Sports Activity",
        ["Regular", "Sometimes", "No"]
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_button = st.button(
        "🔮 Predict Student Grade",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    student_data = {

        "Student_ID": student_id,

        "Student_Age": student_age,

        "Sex": sex,

        "High_School_Type": high_school_type,

        "Scholarship": scholarship,

        "Additional_Work": additional_work,

        "Sports_activity": sports_activity,

        "Transportation": transportation,

        "Weekly_Study_Hours": weekly_study_hours,

        "Attendance": attendance,

        "Reading": reading,

        "Notes": notes,

        "Listening_in_Class": listening,

        "Project_work": project_work
    }

    try:

        with st.spinner("Analyzing student data..."):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=student_data
            )

        if response.status_code == 200:

            result = response.json()

            grade = result["Predicted_Grade"]

            # ==========================================
            # Grade Interpretation
            # ==========================================

            grade_info = {
                "FF": {
                    "score": 0,
                    "level": "Very Poor",
                    "message": "The student may need significant academic support."
                },
                "DD": {
                    "score": 1,
                    "level": "Poor",
                    "message": "The student should improve study habits and academic engagement."
                },
                "DC": {
                    "score": 2,
                    "level": "Below Average",
                    "message": "There is room for improvement in academic performance."
                },
                "CC": {
                    "score": 3,
                    "level": "Average",
                    "message": "The student is performing at an average level."
                },
                "CB": {
                    "score": 4,
                    "level": "Good",
                    "message": "The student is showing good academic performance."
                },
                "BB": {
                    "score": 5,
                    "level": "Very Good",
                    "message": "The student is performing very well academically."
                },
                "BA": {
                    "score": 6,
                    "level": "Excellent",
                    "message": "The student demonstrates excellent academic performance."
                },
                "AA": {
                    "score": 7,
                    "level": "Outstanding",
                    "message": "The student demonstrates outstanding academic performance."
                }
            }

            info = grade_info.get(
                grade,
                {
                    "score": 0,
                    "level": "Unknown",
                    "message": "No interpretation available."
                }
            )

            st.success("Prediction completed successfully!")

            st.markdown(
                f"""
                <div style="
                    padding: 30px;
                    border-radius: 20px;
                    text-align: center;
                    border: 2px solid #4CAF50;
                    margin-top: 20px;
                ">
                    <h2>🎓 Predicted Student Grade</h2>
                    <h1 style="font-size: 60px;">{grade}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ==========================================
            # Performance Information
            # ==========================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Performance Level",
                    info["level"]
                )

            with col2:

                st.metric(
                    "Grade Score",
                    f'{info["score"]} / 7'
                )

            st.info(
                f'💡 {info["message"]}'
            )
            # ==========================================
        # Performance Score
        # ==========================================

            st.markdown("### 📊 Performance Score")

            score = info["score"]

            st.progress(score / 7)

            st.caption(
                f"Performance score: {score} / 7"
            )
            # ==========================================
            # Student Profile
            # ==========================================

            st.markdown("### 👤 Student Profile")

            profile_col1, profile_col2, profile_col3 = st.columns(3)

            with profile_col1:

                st.metric(
                    "📚 Weekly Study",
                    f"{weekly_study_hours} hours"
                )

            with profile_col2:

                st.metric(
                    "📅 Attendance",
                    attendance
                )

            with profile_col3:

                st.metric(
                    "⚽ Sports Activity",
                    sports_activity
                )
                # ==========================================
    # Academic Habits
    # ==========================================

            st.markdown("### 📖 Academic Habits")

            habit_col1, habit_col2, habit_col3, habit_col4 = st.columns(4)

            with habit_col1:

                st.metric(
                    "Reading",
                    reading
                )

            with habit_col2:

                st.metric(
                    "Notes",
                    notes
                )

            with habit_col3:

                st.metric(
                    "Listening",
                    listening
                )

            with habit_col4:

                st.metric(
                    "Project Work",
                    project_work
                )         
        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to the FastAPI server."
        )

        st.info(
            "Make sure FastAPI is running with: "
            "`python -m uvicorn api.app:app --reload`"
        )