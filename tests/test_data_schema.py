"""Guards against silent schema drift in the training data — these are the
columns src/models/train.py hardcodes, so a renamed/missing column should
fail fast here rather than surface as a confusing sklearn error later."""
from __future__ import annotations

from src.models.train import NUMERIC_FEATURES, ONEHOT_FEATURES, ORDINAL_FEATURES

EXPECTED_COLUMNS = {
    "property_type", "sector", "price", "bedRoom", "bathroom", "balcony",
    "agePossession", "built_up_area", "servant room", "store room",
    "furnishing_type", "luxury_category", "floor_category",
}


def test_sample_fixture_has_expected_columns(sample_training_df):
    assert EXPECTED_COLUMNS.issubset(set(sample_training_df.columns))


def test_sample_fixture_has_no_missing_price(sample_training_df):
    assert sample_training_df["price"].isna().sum() == 0


def test_sample_fixture_price_is_positive(sample_training_df):
    assert (sample_training_df["price"] > 0).all()


def test_train_feature_lists_are_subset_of_columns(sample_training_df):
    columns = set(sample_training_df.columns)
    for feature in NUMERIC_FEATURES + ORDINAL_FEATURES + ONEHOT_FEATURES:
        assert feature in columns, f"{feature} referenced by train.py but missing from data"


def test_furnishing_type_is_raw_numeric_before_preprocessing(sample_training_df):
    # train.load_training_data() is responsible for mapping 0/1/2 -> labels;
    # the raw fixture (like the raw Snowflake table) should still be numeric.
    assert sample_training_df["furnishing_type"].dropna().isin([0.0, 1.0, 2.0]).all()
