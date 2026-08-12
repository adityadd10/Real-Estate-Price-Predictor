"""FastAPI serving layer.

The Streamlit UI never touches Snowflake, a pickle, or the model directly —
it only calls this API. That keeps credentials out of the frontend container
and means "the model" can change (retrained, swapped, rolled back) without
touching the UI at all.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.data import analytics_repo
from src.models import predict, recommend
from src.schemas import (
    ModelMetadata,
    NearbyItem,
    PredictionOptions,
    PredictionRequest,
    PredictionResponse,
    RecommendationItem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up: loading (or training) the model...")
    predict.get_model()
    logger.info("Ready.")
    yield


app = FastAPI(title="Real Estate Intelligence API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metadata/options", response_model=PredictionOptions)
def prediction_options():
    return predict.get_options()


@app.get("/metadata/model", response_model=ModelMetadata)
def model_metadata():
    meta = predict.get_metadata()
    return ModelMetadata(
        model_type=meta["model_type"],
        trained_at=meta["trained_at"],
        r2_cv_mean=meta["r2_cv_mean"],
        mae=meta["mae"],
        git_sha=meta.get("git_sha"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        return predict.predict_price(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/recommend/options")
def recommend_options() -> dict:
    return {"properties": recommend.list_properties(), "landmarks": recommend.list_landmarks()}


@app.get("/recommend/similar", response_model=list[RecommendationItem])
def recommend_similar(property: str = Query(...), top_n: int = Query(10, ge=1, le=50)):
    try:
        return recommend.recommend_similar(property, top_n)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown property: {property}") from exc


@app.get("/recommend/nearby", response_model=list[NearbyItem])
def recommend_nearby(landmark: str = Query(...), radius_km: float = Query(..., gt=0)):
    return recommend.nearby_properties(landmark, radius_km)


@app.get("/analytics/heatmap")
def analytics_heatmap() -> list[dict]:
    df = analytics_repo.get_analytics_df()
    grouped = df.groupby("sector")[["price", "price_per_sqft", "built_up_area", "latitude", "longitude"]].mean()
    return grouped.reset_index().to_dict(orient="records")


@app.get("/analytics/scatter")
def analytics_scatter(property_type: str = Query("flat")) -> list[dict]:
    df = analytics_repo.get_analytics_df()
    filtered = df[df["property_type"] == property_type]
    return filtered[["built_up_area", "price", "bedRoom", "sector"]].to_dict(orient="records")


@app.get("/analytics/bhk-distribution")
def analytics_bhk_distribution(sector: str = Query("overall")) -> list[dict]:
    df = analytics_repo.get_analytics_df()
    subset = df if sector == "overall" else df[df["sector"] == sector]
    counts = subset["bedRoom"].value_counts().reset_index()
    counts.columns = ["bedRoom", "count"]
    return counts.to_dict(orient="records")


@app.get("/analytics/price-range")
def analytics_price_range() -> list[dict]:
    df = analytics_repo.get_analytics_df()
    subset = df[df["bedRoom"] <= 4]
    return subset[["bedRoom", "price"]].to_dict(orient="records")


@app.get("/analytics/price-distribution")
def analytics_price_distribution() -> dict:
    df = analytics_repo.get_analytics_df()
    return {
        "house": df.loc[df["property_type"] == "house", "price"].tolist(),
        "flat": df.loc[df["property_type"] == "flat", "price"].tolist(),
    }


@app.get("/analytics/wordcloud")
def analytics_wordcloud(sector: str | None = Query(None)) -> dict:
    import ast

    wc_df = analytics_repo.get_wordcloud_df()
    if not sector:
        words: list[str] = []
        for item in wc_df["features"].dropna().apply(ast.literal_eval):
            words.extend(item)
        return {"text": " ".join(words)}

    filtered = wc_df[wc_df["sector"] == sector]
    words = []
    for item in filtered["features"].dropna().apply(ast.literal_eval):
        words.extend(item)
    return {"text": " ".join(words)}


@app.get("/analytics/sectors")
def analytics_sectors() -> list[str]:
    df = analytics_repo.get_analytics_df()
    return sorted(df["sector"].unique().tolist())
