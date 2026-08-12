import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from api_client import get_prediction_options, predict_price

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="Price Predictor", layout="wide")

# ------------------ LOAD OPTIONS FROM API ------------------ #
try:
    options = get_prediction_options()
except Exception as exc:
    st.error(f"Could not reach the prediction API. Is it running? ({exc})")
    st.stop()

# ------------------ CSS ------------------ #
st.markdown("""
<style>
.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #1C1F26;
    text-align: center;
}
.result-text {
    font-size: 28px;
    font-weight: 600;
    color: #00FFAA;
}
.sub-text {
    font-size: 16px;
    color: #A0A0A0;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.markdown("## 🏠 Property Price Prediction")
st.caption("Enter property details to estimate market price")

# ------------------ FORM ------------------ #
col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown("### 📍 Basic Details")

    property_type = st.selectbox('Property Type', options['property_type'])
    sector = st.selectbox('Sector', options['sector'])
    built_up_area = float(st.number_input('Built Up Area (sq.ft)', min_value=100.0))

    st.markdown("### 🏢 Property Configuration")

    bedrooms = float(st.selectbox('Bedrooms', options['bedroom']))
    bathrooms = float(st.selectbox('Bathrooms', options['bathroom']))
    balcony = st.selectbox('Balconies', options['balcony'])

# RIGHT SIDE
with col2:
    st.markdown("### 🧱 Property Features")

    property_age = st.selectbox('Property Age', options['agePossession'])
    furnishing_type = st.selectbox('Furnishing', options['furnishing_type'])
    floor_category = st.selectbox('Floor Category', options['floor_category'])

    st.markdown("### ⭐ Additional Features")

    luxury_category = st.selectbox('Luxury Category', options['luxury_category'])
    servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
    store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

# ------------------ BUTTON ------------------ #
st.markdown("---")

center = st.columns([1, 2, 1])
with center[1]:
    predict_clicked = st.button("🚀 Predict Price", use_container_width=True)

# ------------------ PREDICTION ------------------ #
if predict_clicked:
    payload = {
        "property_type": property_type,
        "sector": sector,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "balcony": balcony,
        "property_age": property_age,
        "built_up_area": built_up_area,
        "servant_room": servant_room,
        "store_room": store_room,
        "furnishing_type": furnishing_type,
        "luxury_category": luxury_category,
        "floor_category": floor_category,
    }

    with st.spinner("Calling prediction API..."):
        try:
            result = predict_price(payload)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    # ------------------ RESULT DISPLAY ------------------ #
    st.markdown("### 💰 Estimated Price")

    st.markdown(f"""
    <div class="result-box">
        <div class="result-text">₹ {result['low_price_cr']} Cr — ₹ {result['high_price_cr']} Cr</div>
        <div class="sub-text">for a {property_type.capitalize()} in {sector} · model {result['model_version']}</div>
    </div>
    """, unsafe_allow_html=True)
