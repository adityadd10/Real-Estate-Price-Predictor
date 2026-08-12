import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from api_client import (
    get_bhk_distribution,
    get_heatmap_data,
    get_price_distribution,
    get_price_range,
    get_scatter_data,
    get_sectors,
    get_wordcloud_text,
)
from wordcloud import WordCloud

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# ------------------ CSS ------------------ #
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.section {
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.markdown("## 📊 Real Estate Analytics Dashboard")
st.caption("Explore price trends, distributions, and property insights — all served from Snowflake via the API")

try:
    sectors = get_sectors()
except Exception as exc:
    st.error(f"Could not reach the analytics API. Is it running? ({exc})")
    st.stop()

# ------------------ SECTION 1: MAP ------------------ #
st.markdown("### 🌍 Price Heatmap by Sector")

heatmap_df = pd.DataFrame(get_heatmap_data())

fig = px.scatter_map(
    heatmap_df,
    lat="latitude",
    lon="longitude",
    color="price_per_sqft",
    size="built_up_area",
    hover_name="sector",
    zoom=10,
    height=500,
    color_continuous_scale="Turbo"
)

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

st.plotly_chart(fig, use_container_width=True)

# ------------------ SECTION 2: WORDCLOUD ------------------ #
st.markdown("### ☁️ Feature Wordcloud")

col1, col2 = st.columns([1, 2])

with col1:
    sector = st.selectbox('Select Sector', sectors)

features = get_wordcloud_text(sector)

if features.strip():
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#0E1117',
        colormap='viridis'
    ).generate(features)

    fig_wc, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    with col2:
        st.pyplot(fig_wc)
else:
    with col2:
        st.info("No feature text available for this sector.")

# ------------------ SECTION 3: AREA VS PRICE ------------------ #
st.markdown("### 📐 Area vs Price Analysis")

col1, col2 = st.columns([1, 3])

with col1:
    property_type = st.selectbox('Property Type', ['flat', 'house'])

scatter_df = pd.DataFrame(get_scatter_data(property_type))

fig1 = px.scatter(
    scatter_df,
    x="built_up_area",
    y="price",
    color="bedRoom",
    size="price",
    hover_data=["sector"],
    height=450
)

fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0))

with col2:
    st.plotly_chart(fig1, use_container_width=True)

# ------------------ SECTION 4: PIE + BOX ------------------ #
st.markdown("### 🏘️ BHK Distribution & Price Range")

col1, col2 = st.columns(2)

# PIE
sector_option = ['overall'] + sectors
selected_sector = col1.selectbox('Select Sector', sector_option)

bhk_df = pd.DataFrame(get_bhk_distribution(selected_sector))

fig2 = px.pie(
    bhk_df,
    names='bedRoom',
    values='count',
    hole=0.4
)

fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))

col1.plotly_chart(fig2, use_container_width=True)

# BOX
price_range_df = pd.DataFrame(get_price_range())

fig3 = px.box(
    price_range_df,
    x='bedRoom',
    y='price',
    points="all"
)

fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0))

col2.plotly_chart(fig3, use_container_width=True)

# ------------------ SECTION 5: DISTRIBUTION ------------------ #
st.markdown("### 📊 Price Distribution Comparison")

price_dist = get_price_distribution()

fig4, ax = plt.subplots(figsize=(10, 4))

sns.kdeplot(price_dist['house'], label='House', ax=ax)
sns.kdeplot(price_dist['flat'], label='Flat', ax=ax)

ax.legend()
ax.set_xlabel("Price")

st.pyplot(fig4)
