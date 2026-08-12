import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import streamlit as st
from api_client import API_URL, get_model_metadata

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Real Estate Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
<style>
/* Background */
.main {
    background-color: #0E1117;
}

/* Title */
.title {
    font-size: 48px;
    font-weight: 700;
    color: #FFFFFF;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: #A0A0A0;
    margin-bottom: 30px;
}

/* Feature cards */
.card {
    background-color: #1C1F26;
    padding: 25px;
    border-radius: 15px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    background-color: #262A33;
}

/* Card title */
.card-title {
    font-size: 20px;
    font-weight: 600;
    color: #FFFFFF;
}

/* Card text */
.card-text {
    font-size: 14px;
    color: #CFCFCF;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HERO SECTION ------------------ #
st.markdown('<div class="title">🏠 Real Estate Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict prices, explore market trends, and get smart property recommendations — all in one place.</div>',
    unsafe_allow_html=True
)

# ------------------ SERVICE STATUS ------------------ #
try:
    resp = requests.get(f"{API_URL}/health", timeout=5)
    api_up = resp.ok
except requests.RequestException:
    api_up = False

status_col1, status_col2 = st.columns([1, 4])
with status_col1:
    if api_up:
        st.success("API: online")
    else:
        st.error("API: unreachable")
with status_col2:
    if api_up:
        try:
            meta = get_model_metadata()
            st.caption(
                f"Serving `{meta['model_type']}` (git {meta.get('git_sha') or 'local'}) · "
                f"CV R² {meta['r2_cv_mean']:.3f} · MAE {meta['mae']:.3f} Cr · trained {meta['trained_at']}"
            )
        except Exception:
            st.caption(f"Connected to {API_URL}")

# ------------------ FEATURE SECTION ------------------ #
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 Price Prediction</div>
        <div class="card-text">
        ML-powered regression models to accurately predict property prices based on features like location, area, and amenities.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 Market Analytics</div>
        <div class="card-text">
        Interactive visualizations including maps, distributions, and insights to understand real estate trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 Smart Recommendations</div>
        <div class="card-text">
        Personalized property recommendations based on user preferences like budget, location, and amenities.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------ DIVIDER ------------------ #
st.markdown("---")

# ------------------ ABOUT SECTION ------------------ #
st.subheader("🚀 About the Project")

st.write("""
This end-to-end MLOps project integrates **Snowflake (data warehouse), MLflow (experiment
tracking), FastAPI (model serving), Streamlit (UI), Docker (containerization) and GitHub
Actions (CI/CD)** into a single deployable platform.

✔ Training pipeline reads feature-engineered data from Snowflake and logs every run to MLflow
✔ Predictions and recommendations are served by a versioned FastAPI backend, not loaded ad-hoc in the UI
✔ Every push is linted, tested, containerized and published to a container registry automatically

Navigate through the sidebar to explore different modules.
""")
