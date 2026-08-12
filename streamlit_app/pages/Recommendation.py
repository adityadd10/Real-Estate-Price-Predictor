import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from api_client import get_recommend_options, recommend_nearby, recommend_similar

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="Smart Recommendations", layout="wide")

# ------------------ LOAD OPTIONS FROM API ------------------ #
try:
    options = get_recommend_options()
except Exception as exc:
    st.error(f"Could not reach the recommendation API. Is it running? ({exc})")
    st.stop()

# ------------------ CSS ------------------ #
st.markdown("""
<style>
.card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.title {
    font-size: 18px;
    font-weight: 600;
}
.sub {
    font-size: 14px;
    color: #A0A0A0;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.markdown("## 🤖 Smart Property Recommender")
st.caption("Find similar properties and explore nearby options")

# ------------------ LAYOUT ------------------ #
tab1, tab2 = st.tabs(["📍 Nearby Search", "🏠 Similar Properties"])

# ================== TAB 1 ================== #
with tab1:
    st.markdown("### 📍 Find Properties by Location")

    col1, col2 = st.columns(2)

    with col1:
        landmark = st.selectbox('Select Location', options['landmarks'])
    with col2:
        radius = st.number_input('Radius (in KM)', min_value=1.0)

    if st.button('🔍 Search Nearby'):
        with st.spinner("Querying Snowflake..."):
            try:
                results = recommend_nearby(landmark, radius)
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                results = []

        st.markdown("### 📌 Nearby Properties")

        for item in results:
            st.markdown(f"""
            <div class="card">
                <div class="title">{item['property']}</div>
                <div class="sub">Distance: {item['distance_km']} km</div>
            </div>
            """, unsafe_allow_html=True)

# ================== TAB 2 ================== #
with tab2:
    st.markdown("### 🏠 Get Similar Property Recommendations")

    selected_property = st.selectbox('Select Property', options['properties'])

    if st.button('✨ Recommend'):
        with st.spinner("Scoring similarity..."):
            try:
                results = recommend_similar(selected_property)
            except Exception as exc:
                st.error(f"Recommendation failed: {exc}")
                results = []

        st.markdown("### 🔥 Top Recommendations")

        for item in results:
            st.markdown(f"""
            <div class="card">
                <div class="title">{item['property']}</div>
                <div class="sub">Similarity Score: {item['score']}</div>
            </div>
            """, unsafe_allow_html=True)
