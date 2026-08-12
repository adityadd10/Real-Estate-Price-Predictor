"""Central configuration, sourced from environment variables (.env in local dev).

Nothing in this module hardcodes a machine-specific path or a secret — every
value that differs between a laptop, CI, and Render comes from the
environment. See .env.example for the full list of variables.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/config.py -> src -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Snowflake ---
    snowflake_account: str | None = None
    snowflake_user: str | None = None
    snowflake_password: str | None = None
    snowflake_role: str = "ACCOUNTADMIN"
    snowflake_warehouse: str = "REAL_ESTATE_WH"
    snowflake_database: str = "REAL_ESTATE"
    snowflake_schema: str = "GURGAON"

    # --- MLflow ---
    mlflow_tracking_uri: str = f"file:{REPO_ROOT / 'mlruns'}"
    mlflow_experiment_name: str = "gurgaon-price-prediction"

    # --- Artifacts ---
    artifact_dir: Path = REPO_ROOT / "artifacts"
    recommendation_artifact_dir: Path = REPO_ROOT / "artifacts" / "recommendation"
    raw_data_dir: Path = REPO_ROOT / "data" / "raw"

    # --- Service wiring ---
    api_url: str = "http://localhost:8000"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def snowflake_configured(self) -> bool:
        return bool(self.snowflake_account and self.snowflake_user and self.snowflake_password)


settings = Settings()
