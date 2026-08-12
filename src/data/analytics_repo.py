"""Read access for the Analytics dashboard.

Pulls PROPERTIES_ANALYTICS / WORDCLOUD_FEATURES from Snowflake once per process
and caches them in memory (this dataset is small — a few thousand rows — so a
process-lifetime cache is the right tradeoff over re-querying per request).
Falls back to the local CSVs when Snowflake isn't configured, so the API and
its tests still work offline.
"""
from __future__ import annotations

import logging
import threading

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_analytics_df: pd.DataFrame | None = None
_wordcloud_df: pd.DataFrame | None = None


def _load(table: str, csv_name: str) -> pd.DataFrame:
    if settings.snowflake_configured:
        try:
            from src.data.snowflake_client import read_table

            return read_table(table)
        except Exception:
            logger.exception("Failed to read %s from Snowflake, falling back to local CSV", table)
    return pd.read_csv(settings.raw_data_dir / csv_name)


def get_analytics_df() -> pd.DataFrame:
    global _analytics_df
    if _analytics_df is None:
        with _lock:
            if _analytics_df is None:
                _analytics_df = _load("PROPERTIES_ANALYTICS", "data_viz1.csv")
    return _analytics_df


def get_wordcloud_df() -> pd.DataFrame:
    global _wordcloud_df
    if _wordcloud_df is None:
        with _lock:
            if _wordcloud_df is None:
                _wordcloud_df = _load("WORDCLOUD_FEATURES", "wordcloud.csv")
    return _wordcloud_df
