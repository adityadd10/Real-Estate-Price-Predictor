"""Recommendation + nearby-search logic.

Two data sources, deliberately different:

- Similar-property recommendation uses the precomputed cosine-similarity
  matrices under artifacts/recommendation/. These were built in the original
  project's EDA notebook (not part of this repo) and have no reproducible
  source script, so they're versioned as static artifacts rather than
  regenerated. The row order of location_distance.pkl IS the index the
  matrices are aligned to, so this loader must not re-sort or re-fetch that
  ordering from anywhere else.

- Nearby-property search queries Snowflake's LOCATION_DISTANCE table live,
  per request (falls back to the local artifact if Snowflake isn't
  configured, e.g. running tests/CI without credentials). This is the
  "online" counterpart to the training pipeline's batch reads.
"""
from __future__ import annotations

import logging
import pickle
import threading

import pandas as pd

from src.config import settings
from src.schemas import NearbyItem, RecommendationItem

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_location_df: pd.DataFrame | None = None
_cosine_sim = None


def _load_recommendation_artifacts() -> None:
    global _location_df, _cosine_sim
    if _location_df is not None:
        return
    with _lock:
        if _location_df is not None:
            return
        art_dir = settings.recommendation_artifact_dir
        with open(art_dir / "location_distance.pkl", "rb") as f:
            _location_df = pickle.load(f)
        with open(art_dir / "cosine_sim1.pkl", "rb") as f:
            sim1 = pickle.load(f)
        with open(art_dir / "cosine_sim2.pkl", "rb") as f:
            sim2 = pickle.load(f)
        with open(art_dir / "cosine_sim3.pkl", "rb") as f:
            sim3 = pickle.load(f)
        _cosine_sim = 30 * sim1 + 20 * sim2 + 8 * sim3
        logger.info("Loaded recommendation artifacts: %d properties", len(_location_df))


def list_properties() -> list[str]:
    _load_recommendation_artifacts()
    return sorted(_location_df.index.tolist())


def list_landmarks() -> list[str]:
    _load_recommendation_artifacts()
    return sorted(_location_df.columns.tolist())


def recommend_similar(property_name: str, top_n: int = 10) -> list[RecommendationItem]:
    _load_recommendation_artifacts()
    idx = _location_df.index.get_loc(property_name)
    sim_scores = list(enumerate(_cosine_sim[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top = sim_scores[1 : top_n + 1]  # skip self-match at rank 0
    return [
        RecommendationItem(property=_location_df.index[i], score=round(float(score), 4))
        for i, score in top
    ]


def nearby_properties(landmark: str, radius_km: float) -> list[NearbyItem]:
    radius_m = radius_km * 1000

    if settings.snowflake_configured:
        try:
            from src.data.snowflake_client import query

            # Columns were written with lowercase names quoted (see write_table),
            # so they must be quoted here too -- unquoted identifiers fold to
            # UPPERCASE in Snowflake and wouldn't match.
            result = query(
                'SELECT "property", "distance_m" FROM "LOCATION_DISTANCE" '
                'WHERE "landmark" = %s AND "distance_m" < %s ORDER BY "distance_m"',
                (landmark, radius_m),
            )
            return [
                NearbyItem(property=row.property, distance_km=round(row.distance_m / 1000, 2))
                for row in result.itertuples()
            ]
        except Exception:
            logger.exception("Snowflake nearby-search query failed, falling back to local artifact")

    _load_recommendation_artifacts()
    col = _location_df[landmark].dropna()
    col = col[col < radius_m].sort_values()
    return [NearbyItem(property=name, distance_km=round(dist / 1000, 2)) for name, dist in col.items()]
