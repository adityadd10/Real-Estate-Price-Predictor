"""Shared test fixtures.

Tests never touch real Snowflake or the full dataset — they run against
tests/fixtures/sample_properties.csv (a 60-row sample of the real training
data) and a model trained on it in a throwaway artifact directory, so the
whole suite runs offline in CI with no secrets.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_training_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "sample_properties.csv")


@pytest.fixture()
def isolated_artifact_dir(tmp_path, monkeypatch):
    """Point src.config.settings at a scratch dir + a correctly-named copy of the
    sample fixture as the 'raw data' CSV, so training in tests never touches real
    artifacts or Snowflake."""
    import shutil

    from src.config import settings

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(
        FIXTURES_DIR / "sample_properties.csv",
        raw_dir / "gurgaon_properties_post_feature_selection_v2.csv",
    )
    shutil.copy(FIXTURES_DIR / "sample_data_viz1.csv", raw_dir / "data_viz1.csv")
    shutil.copy(FIXTURES_DIR / "sample_wordcloud.csv", raw_dir / "wordcloud.csv")

    monkeypatch.setattr(settings, "artifact_dir", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "raw_data_dir", raw_dir)
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(settings, "snowflake_account", None)  # snowflake_configured -> False
    return tmp_path / "artifacts"
