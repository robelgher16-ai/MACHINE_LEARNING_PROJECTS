
"""House Price Predictor - Streamlit interface.

Run from the project root:
    streamlit run app/app.py

UI code lives in this file. All prediction logic stays in src/predict.py.
"""

import sys
from html import escape
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_price  # noqa: E402


HERO_IMAGE = PROJECT_ROOT / "images" / "hero.png"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon=str(HERO_IMAGE) if HERO_IMAGE.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


NAV_PAGES = ["Home", "Predict", "About"]


# ============================================================
# VALID DATASET RANGES / CATEGORIES
# ============================================================

AREA_RANGE = (501, 4999)
YEAR_RANGE = (1900, 2023)

LOCATIONS = [
    "Downtown",
    "Urban",
    "Suburban",
    "Rural",
]

CONDITIONS = [
    "Excellent",
    "Good",
    "Fair",
    "Poor",
]

GARAGE_OPTIONS = [
    "Yes",
    "No",
]


# ============================================================
# STYLES
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>

        :root {
            --hp-accent: #2F5D8A;
            --hp-accent-dark: #274E74;
            --hp-border: rgba(128, 128, 128, 0.28);
            --hp-soft: rgba(128, 128, 128, 0.07);
            --hp-muted: rgba(128, 128, 128, 0.95);
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1080px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        /* ---------- Navigation ---------- */

        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.25rem;
        }

        .hp-nav div[data-testid="stRadio"] label {
            padding: 0.45rem 1.1rem;
            border-radius: 8px;
            border: 1px solid transparent;
            cursor: pointer;
        }

        .hp-nav div[data-testid="stRadio"] label:has(input:checked) {
            background: var(--hp-soft);
            border-color: var(--hp-border);
            font-weight: 600;
        }

        .hp-nav div[data-testid="stRadio"] label > div:first-child {
            display: none;
        }

        /* ---------- Typography ---------- */

        .hp-brand {
            font-weight: 700;
            font-size: 1.05rem;
            letter-spacing: 0.2px;
        }

        .hp-title {
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0.4rem 0 0.6rem 0;
        }

        .hp-subtitle {
            font-size: 1.2rem;
            font-weight: 500;
            color: var(--hp-accent);
            margin-bottom: 1rem;
        }

        .hp-lead {
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 34rem;
        }

        .hp-section {
            font-size: 1.5rem;
            font-weight: 650;
            margin: 2rem 0 0.4rem 0;
        }

        .hp-caption {
            color: var(--hp-muted);
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        /* ---------- Home Cards ---------- */

        .hp-card {
            border: 1px solid var(--hp-border);
            border-radius: 12px;
            padding: 1.3rem 1.4rem;
            box-shadow:
                0 1px 3px rgba(0, 0, 0, 0.06),
                0 6px 18px rgba(0, 0, 0, 0.04);
            height: 100%;
        }

        .hp-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.02rem;
        }

        .hp-card p {
            margin: 0;
            line-height: 1.55;
            font-size: 0.95rem;
        }

        /* ---------- Form ---------- */

        div[data-testid="stForm"] {
            border: 1px solid var(--hp-border);
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }

        button[kind="primary"],
        button[kind="primaryFormSubmit"] {
            background-color: var(--hp-accent);
            border: 1px solid var(--hp-accent);
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1rem;
        }

        button[kind="primary"]:hover,
        button[kind="primaryFormSubmit"]:hover {
            background-color: var(--hp-accent-dark);
            border-color: var(--hp-accent-dark);
        }

        button[kind="secondary"] {
            border-radius: 8px;
            font-weight: 600;
        }

        /* ---------- Responsive ---------- */

        @media (max-width: 768px) {

            .hp-title {
                font-size: 2.2rem;
            }

        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HELPERS
# ============================================================

def go_to(page: str) -> None:
    """Button callback: switch the navigation to `page`."""
    st.session_state["nav"] = page


def reset_prediction() -> None:
    """Button callback: clear the stored result."""
    st.session_state.pop("result", None)


def card(title: str, body: str) -> str:
    return (
        f'<div class="hp-card">'
        f"<h4>{escape(title)}</h4>"
        f"<p>{escape(body)}</p>"
        f"</div>"
    )


# ============================================================
# NAVIGATION
# ============================================================

def render_navigation() -> str:

    brand_col, nav_col = st.columns([1, 1.4])

    with brand_col:
        st.markdown(
            '<div class="hp-brand">House Price Predictor</div>',
            unsafe_allow_html=True,
        )

    with nav_col:

        st.markdown(
            '<div class="hp-nav">',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            NAV_PAGES,
            key="nav",
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    return page


# ============================================================
# FOOTER
# ============================================================

def render_footer() -> None:

    st.markdown(
        """
        <div class="hp-footer">
            <strong>House Price Prediction</strong><br>
            Machine Learning Portfolio Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOME PAGE
# ============================================================

def render_home() -> None:

    text_col, image_col = st.columns(
        [1.15, 1],
        gap="large",
        vertical_alignment="center",
    )

    with text_col:

        st.markdown(
            '<div class="hp-title">House Price Predictor</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="hp-subtitle">'
            "Machine Learning Powered Real Estate Price Prediction"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="hp-lead">'
            "Enter a property's characteristics, such as its size, "
            "number of rooms, age, location, and condition, and the "
            "trained model returns a predicted house price."
            "</div>",
            unsafe_allow_html=True,
        )

        st.write("")

        st.button(
            "Start a prediction",
            type="primary",
            on_click=go_to,
            args=("Predict",),
        )

    with image_col:

        if HERO_IMAGE.exists():
            st.image(
                str(HERO_IMAGE),
                use_container_width=True,
            )

    st.markdown(
        '<div class="hp-section">How it works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hp-caption">'
        "Two groups of inputs, one prediction."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        card(
            "Describe the property",
            "Area, bedrooms, bathrooms, floors, and year built.",
        ),
        unsafe_allow_html=True,
    )

    c2.markdown(
        card(
            "Add the details",
            "Location, overall condition, and whether there is a garage.",
        ),
        unsafe_allow_html=True,
    )

    c3.markdown(
        card(
            "Get an estimate",
            "The saved regression model returns a predicted price.",
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

def render_prediction() -> None:

    st.markdown(
        '<div class="hp-section" style="margin-top:0.4rem">'
        "Predict"
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    if "result" in st.session_state:

        render_result(
            st.session_state["result"]
        )

        return

    # --------------------------------------------------------
    # FORM DESCRIPTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="hp-caption">'
        "Fill in the property details below. "
        "Values are limited to the ranges found in the training dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PREDICTION FORM
    # --------------------------------------------------------

    with st.form(
        "prediction_form",
        border=True,
    ):

        left, right = st.columns(
            2,
            gap="large",
        )

        with left:

            st.markdown("**Property Information**")

            area = st.number_input(
                "Area (sq ft)",
                min_value=AREA_RANGE[0],
                max_value=AREA_RANGE[1],
                value=2500,
                step=50,
            )

            bedrooms = st.radio(
                "Bedrooms",
                [1, 2, 3, 4, 5],
                index=3,
                horizontal=True,
            )

            bathrooms = st.radio(
                "Bathrooms",
                [1, 2, 3, 4],
                index=2,
                horizontal=True,
            )

            floors = st.radio(
                "Floors",
                [1, 2, 3],
                index=1,
                horizontal=True,
            )

            year_built = st.number_input(
                "Year Built",
                min_value=YEAR_RANGE[0],
                max_value=YEAR_RANGE[1],
                value=2018,
                step=1,
            )

        with right:

            st.markdown("**Property Details**")

            location = st.selectbox(
                "Location",
                LOCATIONS,
                index=1,
            )

            condition = st.selectbox(
                "Condition",
                CONDITIONS,
                index=1,
            )

            garage = st.radio(
                "Garage",
                GARAGE_OPTIONS,
                index=0,
                horizontal=True,
            )

        submitted = st.form_submit_button(
            "Predict House Price",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    if submitted:

        inputs = {
            "area": int(area),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "floors": int(floors),
            "year_built": int(year_built),
            "location": location,
            "condition": condition,
            "garage": garage,
        }

        try:

            price = predict_price(
                **inputs
            )

        except Exception as exc:

            st.error(
                f"The prediction could not be completed: {exc}"
            )

            return

        st.session_state["result"] = {
            "price": price,
            "inputs": inputs,
        }

        st.rerun()


# ============================================================
# PREDICTION RESULT
# ============================================================

def render_result(result: dict) -> None:

    price = result["price"]
    inputs = result["inputs"]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.success(
        "Prediction completed successfully."
    )

    st.subheader(
        "Estimated House Price"
    )

    st.metric(
        label="Predicted Price",
        value=f"${price:,.2f}",
    )

    st.caption(
        "Output of the saved Linear Regression model "
        "for the property characteristics below."
    )

    st.divider()

    # --------------------------------------------------------
    # PROPERTY SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Property Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Area",
            f"{inputs['area']:,} sq ft",
        )

    with col2:
        st.metric(
            "Bedrooms",
            inputs["bedrooms"],
        )

    with col3:
        st.metric(
            "Bathrooms",
            inputs["bathrooms"],
        )

    with col4:
        st.metric(
            "Floors",
            inputs["floors"],
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Year Built",
            inputs["year_built"],
        )

    with col2:
        st.metric(
            "Location",
            inputs["location"],
        )

    with col3:
        st.metric(
            "Condition",
            inputs["condition"],
        )

    with col4:
        st.metric(
            "Garage",
            inputs["garage"],
        )

    st.divider()

    st.info(
        "This value is generated by the trained machine learning "
        "model using the property characteristics you provided."
    )

    st.button(
        "Make Another Prediction",
        on_click=reset_prediction,
    )


# ============================================================
# ABOUT PAGE
# ============================================================

def render_about() -> None:

    st.markdown(
        '<div class="hp-section" style="margin-top:0.4rem">'
        "About the Project"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "This application demonstrates an end-to-end machine learning "
        "workflow for house-price prediction. It is a machine-learning "
        "engineering and learning project, not a production valuation tool."
    )

    left, right = st.columns(
        [1.2, 1],
        gap="large",
    )

    with left:

        st.markdown(
            "**What the project covers**"
        )

        st.markdown(
            "- Data preprocessing\n"
            "- Feature engineering\n"
            "- Multiple regression models\n"
            "- Model evaluation\n"
            "- Saved model\n"
            "- Prediction pipeline\n"
            "- Streamlit deployment\n"
            "- FastAPI API"
        )

        st.markdown(
            "**A note on performance**"
        )

        st.markdown(
            "The relationships between the available features and the "
            "target price are weak in this dataset, so the model does not "
            "predict prices well. Treat its output as a demonstration of "
            "the workflow rather than a reliable estimate."
        )

    with right:

        st.markdown(
            "**Model Information**"
        )

        st.write(
            "Model: Linear Regression"
        )

        st.write(
            "Task: Regression"
        )

        st.write(
            "Dataset: 2,000 records"
        )

        st.write(
            "Target: House Price"
        )

        st.write(
            "R²: Approximately -0.0067"
        )

        st.caption(
            "A negative R² means the model performs slightly worse "
            "than predicting the average price."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    inject_css()

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Home"

    page = render_navigation()

    if page == "Home":

        render_home()

    elif page == "Predict":

        render_prediction()

    else:

        render_about()

    render_footer()


main()

