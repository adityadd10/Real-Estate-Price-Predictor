"""Inference-side model loading + prediction.

Loads the artifact produced by src.models.train (artifacts/pipeline.pkl +
model_metadata.json). If the artifact is missing — e.g. a fresh checkout that
hasn't trained yet — it trains once, on first use, from whichever data source
is configured (Snowflake if creds are present, local CSV otherwise). This is
what lets `docker compose up` boot the API from nothing but source + env vars.
"""
from __future__ import annotations

import json
import logging
import threading

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipeline = None
_metadata: dict | None = None

PREDICTION_HALF_WIDTH_CR = 0.22  # matches the original project's naive +/- band


def _ensure_trained() -> None:
    pipeline_path = settings.artifact_dir / "pipeline.pkl"
    metadata_path = settings.artifact_dir / "model_metadata.json"
    if pipeline_path.exists() and metadata_path.exists():
        return
    logger.warning("No trained model artifact found at %s — training now.", pipeline_path)
    from src.models.train import main as train_main

    # argv=[] -- without this, argparse reads sys.argv from the *running
    # process* (e.g. uvicorn's own CLI args) and crashes. See train.main().
    train_main(argv=[])


def get_model():
    global _pipeline, _metadata
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _ensure_trained()
                pipeline_path = settings.artifact_dir / "pipeline.pkl"
                metadata_path = settings.artifact_dir / "model_metadata.json"
                _pipeline = joblib.load(pipeline_path)
                _metadata = json.loads(metadata_path.read_text())
                logger.info("Loaded model %s trained at %s", _metadata["model_type"], _metadata["trained_at"])
    return _pipeline, _metadata


def get_options() -> dict:
    _, metadata = get_model()
    return metadata["options"]


def get_metadata() -> dict:
    _, metadata = get_model()
    return metadata


def predict_price(request: PredictionRequest) -> PredictionResponse:
    pipeline, metadata = get_model()

    row = pd.DataFrame(
        [
            {
                "property_type": request.property_type,
                "sector": request.sector,
                "bedRoom": request.bedrooms,
                "bathroom": request.bathrooms,
                "balcony": request.balcony,
                "agePossession": request.property_age,
                "built_up_area": request.built_up_area,
                "servant room": request.servant_room,
                "store room": request.store_room,
                "furnishing_type": request.furnishing_type,
                "luxury_category": request.luxury_category,
                "floor_category": request.floor_category,
            }
        ]
    )

    point_estimate = float(np.expm1(pipeline.predict(row))[0])

    return PredictionResponse(
        low_price_cr=round(point_estimate - PREDICTION_HALF_WIDTH_CR, 2),
        high_price_cr=round(point_estimate + PREDICTION_HALF_WIDTH_CR, 2),
        point_estimate_cr=round(point_estimate, 2),
        model_version=f"{metadata['model_type']}@{metadata.get('git_sha') or 'local'}",
    )
