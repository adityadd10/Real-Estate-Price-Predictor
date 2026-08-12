"""API-level tests. Uses the same offline fixture data as test_model.py —
the FastAPI lifespan's startup training runs against it via isolated_artifact_dir,
so no Snowflake connection is ever made. Recommendation endpoints use the real
committed artifacts/recommendation/*.pkl (small, static, checked into the repo)."""
from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, isolated_artifact_dir):
    import src.models.predict as predict_module

    predict_module._pipeline = None
    predict_module._metadata = None
    monkeypatch.setattr(sys, "argv", ["pytest"])

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    predict_module._pipeline = None
    predict_module._metadata = None


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metadata_options(client):
    resp = client.get("/metadata/options")
    assert resp.status_code == 200
    body = resp.json()
    assert "sector" in body and len(body["sector"]) > 0


def test_metadata_model(client):
    resp = client.get("/metadata/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "xgboost"


def test_predict_endpoint(client):
    options = client.get("/metadata/options").json()
    payload = {
        "property_type": options["property_type"][0],
        "sector": options["sector"][0],
        "bedrooms": options["bedroom"][0],
        "bathrooms": options["bathroom"][0],
        "balcony": options["balcony"][0],
        "property_age": options["agePossession"][0],
        "built_up_area": 1200,
        "servant_room": 0.0,
        "store_room": 0.0,
        "furnishing_type": options["furnishing_type"][0],
        "luxury_category": options["luxury_category"][0],
        "floor_category": options["floor_category"][0],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["high_price_cr"] > body["low_price_cr"]


def test_predict_endpoint_rejects_malformed_payload(client):
    resp = client.post("/predict", json={"property_type": "flat"})
    assert resp.status_code == 422


def test_recommend_options(client):
    resp = client.get("/recommend/options")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["properties"]) > 0
    assert len(body["landmarks"]) > 0


def test_recommend_similar(client):
    props = client.get("/recommend/options").json()["properties"]
    resp = client.get("/recommend/similar", params={"property": props[0], "top_n": 5})
    assert resp.status_code == 200
    assert len(resp.json()) <= 5


def test_recommend_similar_unknown_property_404s(client):
    resp = client.get("/recommend/similar", params={"property": "does-not-exist-12345"})
    assert resp.status_code == 404


def test_recommend_nearby(client):
    landmarks = client.get("/recommend/options").json()["landmarks"]
    resp = client.get("/recommend/nearby", params={"landmark": landmarks[0], "radius_km": 5})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_analytics_sectors(client):
    resp = client.get("/analytics/sectors")
    assert resp.status_code == 200
    assert len(resp.json()) > 0


def test_analytics_heatmap(client):
    resp = client.get("/analytics/heatmap")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_analytics_wordcloud(client):
    resp = client.get("/analytics/wordcloud")
    assert resp.status_code == 200
    assert "text" in resp.json()
