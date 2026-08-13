"""End-to-end test of the training + inference path, run against the small
offline fixture (no Snowflake, no network)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from src.schemas import PredictionRequest


def _reset_predict_module_cache():
    """predict.py caches the loaded pipeline in module globals; tests that
    train into a fresh tmp_path artifact dir need a clean slate each time."""
    import src.models.predict as predict_module

    predict_module._pipeline = None
    predict_module._metadata = None


@pytest.fixture(autouse=True)
def _reset_cache():
    _reset_predict_module_cache()
    yield
    _reset_predict_module_cache()


def test_load_training_data_maps_furnishing_type(isolated_artifact_dir):
    from src.models.train import load_training_data

    df = load_training_data(source="csv")
    assert set(df["furnishing_type"].unique()) <= {"unfurnished", "semifurnished", "furnished"}


def test_train_final_model_returns_fitted_pipeline_and_metrics(isolated_artifact_dir):
    from src.models.train import load_training_data, train_final_model

    df = load_training_data(source="csv")
    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])

    pipeline, metrics = train_final_model(X, y)

    assert "r2_cv_mean" in metrics and "mae" in metrics
    assert metrics["mae"] >= 0
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_main_writes_artifacts(monkeypatch, isolated_artifact_dir):
    from src.models import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--source", "csv"])
    train.main()

    assert (isolated_artifact_dir / "pipeline.pkl").exists()
    metadata_path = isolated_artifact_dir / "model_metadata.json"
    assert metadata_path.exists()

    import json

    metadata = json.loads(metadata_path.read_text())
    assert metadata["model_type"] == "xgboost"
    assert "options" in metadata
    assert "sector" in metadata["options"]


def test_predict_price_end_to_end(monkeypatch, isolated_artifact_dir):
    import sys as _sys

    from src.models import predict, train

    monkeypatch.setattr(_sys, "argv", ["train.py", "--source", "csv"])
    train.main()

    options = predict.get_options()
    request = PredictionRequest(
        property_type=options["property_type"][0],
        sector=options["sector"][0],
        bedrooms=options["bedroom"][0],
        bathrooms=options["bathroom"][0],
        balcony=options["balcony"][0],
        property_age=options["agePossession"][0],
        built_up_area=1200,
        servant_room=0.0,
        store_room=0.0,
        furnishing_type=options["furnishing_type"][0],
        luxury_category=options["luxury_category"][0],
        floor_category=options["floor_category"][0],
    )

    response = predict.predict_price(request)

    assert response.high_price_cr > response.low_price_cr
    assert response.point_estimate_cr > 0
    assert response.model_version.startswith("xgboost@")


def test_self_training_ignores_host_process_argv(monkeypatch, isolated_artifact_dir):
    """Regression test: src.models.predict._ensure_trained() calls train.main()
    from *inside a running server process* to self-train when no artifact
    exists yet. If that call let argparse read the real sys.argv, it would try
    to parse whatever the host process was actually launched with (e.g.
    uvicorn's own "api.main:app --host 0.0.0.0 --port 8000") and crash the
    entire app at startup -- which is exactly what happened the first time
    this was tested against a genuinely empty Docker volume instead of a
    pre-populated artifact dir. Simulate that real invocation here."""
    import sys as _sys

    from src.models import predict

    monkeypatch.setattr(_sys, "argv", ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

    pipeline, metadata = predict.get_model()

    assert metadata["model_type"] == "xgboost"
    assert pipeline is not None


def test_predict_price_unseen_category_does_not_crash(monkeypatch, isolated_artifact_dir):
    """A sector/agePossession value the model never saw during training should
    be handled gracefully (OrdinalEncoder handle_unknown='use_encoded_value'),
    not raise — this is what happens if Snowflake data drifts after training."""
    import sys as _sys

    from src.models import predict, train

    monkeypatch.setattr(_sys, "argv", ["train.py", "--source", "csv"])
    train.main()

    options = predict.get_options()
    request = PredictionRequest(
        property_type=options["property_type"][0],
        sector="sector 9999-does-not-exist",
        bedrooms=options["bedroom"][0],
        bathrooms=options["bathroom"][0],
        balcony=options["balcony"][0],
        property_age=options["agePossession"][0],
        built_up_area=1200,
        servant_room=0.0,
        store_room=0.0,
        furnishing_type=options["furnishing_type"][0],
        luxury_category=options["luxury_category"][0],
        floor_category=options["floor_category"][0],
    )

    response = predict.predict_price(request)
    assert response.point_estimate_cr > 0
