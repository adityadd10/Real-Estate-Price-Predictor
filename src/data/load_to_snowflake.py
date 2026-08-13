"""One-time (re-runnable) ingestion job: local CSV/pickle artifacts -> Snowflake.

This is the "batch load" side of the pipeline. Run it once after you've
created your Snowflake trial and filled in .env:

    python -m src.data.load_to_snowflake

It creates the warehouse/database/schema if they don't exist, then loads four
tables:

  PROPERTIES_TRAIN     -- feature-engineered training set (used by src/models/train.py)
  PROPERTIES_ANALYTICS -- richer table (lat/lon, price_per_sqft) used by the Analytics page
  WORDCLOUD_FEATURES   -- per-sector feature keyword lists used by the Analytics wordcloud
  LOCATION_DISTANCE    -- long-format (property, landmark, distance_m) table, melted from
                          the original wide property x landmark distance matrix, so the API
                          can do the "nearby search" as a live SQL query instead of a pandas
                          scan.

The recommendation similarity matrices (cosine_sim*.pkl) are NOT loaded here — they're
dense property x property matrices with no reproducible source script (built in the
original project's EDA notebook, which isn't part of this repo). They're versioned as
static model artifacts under artifacts/recommendation/ instead. See README for details.
"""
from __future__ import annotations

import logging
import pickle

import pandas as pd

from src.config import settings
from src.data.snowflake_client import execute, write_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_snowflake_objects() -> None:
    """Create the warehouse/database/schema if they don't already exist.
    Idempotent — safe to run on every deploy, not just the first time."""
    execute(
        f"CREATE WAREHOUSE IF NOT EXISTS {settings.snowflake_warehouse} "
        "WITH WAREHOUSE_SIZE='XSMALL' AUTO_SUSPEND=60 AUTO_RESUME=TRUE INITIALLY_SUSPENDED=TRUE"
    )
    execute(f"CREATE DATABASE IF NOT EXISTS {settings.snowflake_database}")
    execute(
        f"CREATE SCHEMA IF NOT EXISTS {settings.snowflake_database}.{settings.snowflake_schema}"
    )
    logger.info(
        "Warehouse/database/schema ready: %s / %s.%s",
        settings.snowflake_warehouse,
        settings.snowflake_database,
        settings.snowflake_schema,
    )


def load_properties_train() -> None:
    path = settings.raw_data_dir / "gurgaon_properties_post_feature_selection_v2.csv"
    df = pd.read_csv(path)
    write_table(df, "PROPERTIES_TRAIN")
    logger.info("PROPERTIES_TRAIN: %d rows loaded from %s", len(df), path.name)


def load_properties_analytics() -> None:
    path = settings.raw_data_dir / "data_viz1.csv"
    df = pd.read_csv(path)
    write_table(df, "PROPERTIES_ANALYTICS")
    logger.info("PROPERTIES_ANALYTICS: %d rows loaded from %s", len(df), path.name)


def load_wordcloud_features() -> None:
    path = settings.raw_data_dir / "wordcloud.csv"
    df = pd.read_csv(path)
    write_table(df, "WORDCLOUD_FEATURES")
    logger.info("WORDCLOUD_FEATURES: %d rows loaded from %s", len(df), path.name)


def load_location_distance() -> None:
    """Melt the wide property x landmark distance matrix into a long table so it's
    queryable with plain SQL (property, landmark, distance_m)."""
    path = settings.recommendation_artifact_dir / "location_distance.pkl"
    with open(path, "rb") as f:
        wide_df = pickle.load(f)

    # reset_index() names the new column after the index's own name (e.g.
    # "PropertyName") when it has one, and "index" only when it doesn't --
    # handle both rather than assuming the unnamed case.
    index_col = wide_df.index.name or "index"
    long_df = (
        wide_df.reset_index()
        .rename(columns={index_col: "property"})
        .melt(id_vars="property", var_name="landmark", value_name="distance_m")
        .dropna(subset=["distance_m"])
    )
    write_table(long_df, "LOCATION_DISTANCE")
    logger.info(
        "LOCATION_DISTANCE: %d rows loaded (melted from %s x %s matrix)",
        len(long_df),
        wide_df.shape[0],
        wide_df.shape[1],
    )


def main() -> None:
    bootstrap_snowflake_objects()
    load_properties_train()
    load_properties_analytics()
    load_wordcloud_features()
    load_location_distance()
    logger.info("Snowflake ingestion complete.")


if __name__ == "__main__":
    main()
