import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_price


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .info-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate the market price of a house using a trained machine learning model.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# HOUSE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">House Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=1,
        max_value=10000,
        value=2500,
        step=100
    )

    bedrooms = st.selectbox(
        "Bedrooms",
        options=[1, 2, 3, 4, 5],
        index=3
    )

    bathrooms = st.selectbox(
        "Bathrooms",
        options=[1, 2, 3, 4],
        index=2
    )

    floors = st.selectbox(
        "Floors",
        options=[1, 2, 3],
        index=1
    )


with col2:

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2023,
        value=2018,
        step=1
    )

    location = st.selectbox(
        "Location",
        options=[
            "Downtown",
            "Urban",
            "Suburban",
            "Rural"
        ]
    )

    condition = st.selectbox(
        "Condition",
        options=[
            "Excellent",
            "Good",
            "Fair",
            "Poor"
        ]
    )

    garage = st.selectbox(
        "Garage",
        options=[
            "Yes",
            "No"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "Predict House Price",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    price = predict_price(
        area=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        floors=floors,
        year_built=year_built,
        location=location,
        condition=condition,
        garage=garage
    )

    st.success("Prediction completed successfully.")

    st.metric(
        label="Estimated House Price",
        value=f"${price:,.2f}"
    )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.markdown(
    """
    <div class="info-box">

    <strong>About this application</strong><br><br>

    This application uses a trained machine learning model to
    estimate house prices from property characteristics including
    area, bedrooms, bathrooms, floors, year built, location,
    condition, and garage availability.

    </div>
    """,
    unsafe_allow_html=True
)