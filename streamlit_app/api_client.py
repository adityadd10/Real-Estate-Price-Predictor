"""Small requests wrapper the Streamlit pages use to talk to the FastAPI service.

API_URL comes from the environment (docker-compose sets it to the internal
service name; Render/local dev set it to a public or localhost URL). Nothing
here loads a pickle, opens a Snowflake connection, or hardcodes a path — the
UI is a pure client of the API.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=300)
def get_prediction_options() -> dict:
    resp = requests.get(f"{API_URL}/metadata/options", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_model_metadata() -> dict:
    resp = requests.get(f"{API_URL}/metadata/model", timeout=15)
    resp.raise_for_status()
    return resp.json()


def predict_price(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_recommend_options() -> dict:
    resp = requests.get(f"{API_URL}/recommend/options", timeout=15)
    resp.raise_for_status()
    return resp.json()


def recommend_similar(property_name: str, top_n: int = 10) -> list[dict]:
    resp = requests.get(
        f"{API_URL}/recommend/similar", params={"property": property_name, "top_n": top_n}, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def recommend_nearby(landmark: str, radius_km: float) -> list[dict]:
    resp = requests.get(
        f"{API_URL}/recommend/nearby", params={"landmark": landmark, "radius_km": radius_km}, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_heatmap_data() -> list[dict]:
    resp = requests.get(f"{API_URL}/analytics/heatmap", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_scatter_data(property_type: str) -> list[dict]:
    resp = requests.get(f"{API_URL}/analytics/scatter", params={"property_type": property_type}, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_bhk_distribution(sector: str) -> list[dict]:
    resp = requests.get(f"{API_URL}/analytics/bhk-distribution", params={"sector": sector}, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_price_range() -> list[dict]:
    resp = requests.get(f"{API_URL}/analytics/price-range", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_price_distribution() -> dict:
    resp = requests.get(f"{API_URL}/analytics/price-distribution", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_wordcloud_text(sector: str | None) -> str:
    resp = requests.get(f"{API_URL}/analytics/wordcloud", params={"sector": sector} if sector else {}, timeout=15)
    resp.raise_for_status()
    return resp.json()["text"]


@st.cache_data(ttl=300)
def get_sectors() -> list[str]:
    resp = requests.get(f"{API_URL}/analytics/sectors", timeout=15)
    resp.raise_for_status()
    return resp.json()
