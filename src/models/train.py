"""Training entrypoint.

    python -m src.models.train                 # train the production model, log to MLflow
    python -m src.models.train --compare        # also run the 11-model CV bake-off first
    python -m src.models.train --source csv      # skip Snowflake, use data/raw/*.csv

Reads the feature-engineered training table (Snowflake by default, local CSV as a
fallback for offline dev/CI), fits the same preprocessing pipeline as the original
project, and replaces the original 500-tree RandomForest (a 140MB artifact) with a
tuned XGBoost model that scores comparably on cross-validated R2/MAE at roughly 1%
of the file size — the difference between "commit the model to git" and "must be
fetched from somewhere else" for a Docker image.

Every run is logged to MLflow (params, CV metrics, the fitted pipeline) and, if
Snowflake is configured, appended as a row to MODEL_RUNS for lineage/governance.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["bedRoom", "bathroom", "built_up_area", "servant room", "store room"]
ORDINAL_FEATURES = [
    "property_type",
    "sector",
    "balcony",
    "agePossession",
    "furnishing_type",
    "luxury_category",
    "floor_category",
]
ONEHOT_FEATURES = ["sector", "agePossession"]


def git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_training_data(source: str) -> pd.DataFrame:
    if source == "snowflake":
        from src.data.snowflake_client import read_table

        logger.info("Loading PROPERTIES_TRAIN from Snowflake...")
        df = read_table("PROPERTIES_TRAIN")
    else:
        path = settings.raw_data_dir / "gurgaon_properties_post_feature_selection_v2.csv"
        logger.info("Loading training data from local CSV fallback: %s", path)
        df = pd.read_csv(path)

    df["furnishing_type"] = df["furnishing_type"].replace(
        {0.0: "unfurnished", 1.0: "semifurnished", 2.0: "furnished"}
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ORDINAL_FEATURES,
            ),
            ("cat1", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ONEHOT_FEATURES),
        ],
        remainder="passthrough",
    )


def cv_bakeoff(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Reproduces the original project's model comparison. Off by default (slow:
    11 models x 10-fold CV) — pass --compare to run it and see where XGBoost lands."""
    candidates = {
        "linear_reg": LinearRegression(),
        "svr": SVR(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "adaboost": AdaBoostRegressor(random_state=42),
        "mlp": MLPRegressor(max_iter=500, random_state=42),
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
        ),
    }
    preprocessor = build_preprocessor()
    rows = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
        r2 = cross_val_score(pipe, X, y, cv=kfold, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(pipe.predict(X_test)))
        rows.append({"model": name, "r2_cv_mean": r2, "mae": mae})
        logger.info("  %-18s r2=%.4f  mae=%.4f", name, r2, mae)
    result = pd.DataFrame(rows).sort_values("mae")
    logger.info("Bake-off results (best MAE first):\n%s", result.to_string(index=False))
    return result


def train_final_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    preprocessor = build_preprocessor()
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kfold, scoring="r2").mean()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(pipeline.predict(X_test)))

    # Refit on all data for the artifact that actually ships.
    pipeline.fit(X, y)

    metrics = {"r2_cv_mean": float(cv_r2), "mae": float(mae)}
    return pipeline, metrics


def build_prediction_options(df: pd.DataFrame) -> dict:
    return {
        "property_type": sorted(df["property_type"].dropna().unique().tolist()),
        "sector": sorted(df["sector"].dropna().unique().tolist()),
        "bedroom": sorted(df["bedRoom"].dropna().unique().tolist()),
        "bathroom": sorted(df["bathroom"].dropna().unique().tolist()),
        "balcony": sorted(df["balcony"].dropna().unique().tolist()),
        "agePossession": sorted(df["agePossession"].dropna().unique().tolist()),
        "furnishing_type": sorted(df["furnishing_type"].dropna().unique().tolist()),
        "luxury_category": sorted(df["luxury_category"].dropna().unique().tolist()),
        "floor_category": sorted(df["floor_category"].dropna().unique().tolist()),
    }


def log_run_to_snowflake(metrics: dict, model_type: str, sha: str | None) -> None:
    if not settings.snowflake_configured:
        logger.info("Snowflake not configured — skipping MODEL_RUNS lineage write.")
        return
    try:
        from src.data.snowflake_client import write_table

        run_row = pd.DataFrame(
            [
                {
                    "trained_at": datetime.now(UTC).isoformat(),
                    "model_type": model_type,
                    "r2_cv_mean": metrics["r2_cv_mean"],
                    "mae": metrics["mae"],
                    "git_sha": sha or "unknown",
                }
            ]
        )
        write_table(run_row, "MODEL_RUNS", overwrite=False)
    except Exception:
        logger.exception("Failed to write MODEL_RUNS row to Snowflake (non-fatal)")


def main(argv: list[str] | None = None) -> None:
    """argv defaults to None, which makes argparse read sys.argv[1:] -- correct
    when this runs as a script. src.models.predict calls this programmatically
    from inside a running uvicorn process to self-train on first boot, and
    must pass argv=[] there instead, or argparse would try to parse uvicorn's
    own CLI arguments and crash the whole app at startup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["snowflake", "csv"], default="snowflake" if settings.snowflake_configured else "csv")
    parser.add_argument("--compare", action="store_true", help="Run the full model bake-off before training the final model")
    args = parser.parse_args(argv)

    df = load_training_data(args.source)
    X = df.drop(columns=["price"])
    y_transformed = np.log1p(df["price"])

    if args.compare:
        logger.info("Running model bake-off (--compare)...")
        cv_bakeoff(X, y_transformed)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    sha = git_sha()
    with mlflow.start_run():
        pipeline, metrics = train_final_model(X, y_transformed)

        mlflow.log_params(
            {
                "model_type": "xgboost",
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "data_source": args.source,
                "git_sha": sha or "unknown",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = settings.artifact_dir / "pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)

        metadata = {
            "model_type": "xgboost",
            "trained_at": datetime.now(UTC).isoformat(),
            "git_sha": sha,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            **metrics,
            "options": build_prediction_options(df),
        }
        metadata_path = settings.artifact_dir / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        logger.info("Saved pipeline -> %s (%.1f MB)", pipeline_path, pipeline_path.stat().st_size / 1e6)
        logger.info("R2 (10-fold CV): %.4f | MAE: %.4f Cr", metrics["r2_cv_mean"], metrics["mae"])

    log_run_to_snowflake(metrics, "xgboost", sha)


if __name__ == "__main__":
    main()
