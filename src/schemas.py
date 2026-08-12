"""Pydantic request/response models shared by the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    property_type: str = Field(..., examples=["flat"])
    sector: str = Field(..., examples=["sector 102"])
    bedrooms: float = Field(..., ge=1, examples=[3])
    bathrooms: float = Field(..., ge=1, examples=[3])
    balcony: str = Field(..., examples=["3+"])
    property_age: str = Field(..., examples=["New Property"])
    built_up_area: float = Field(..., gt=0, examples=[1450])
    servant_room: float = Field(..., examples=[1.0])
    store_room: float = Field(..., examples=[0.0])
    furnishing_type: str = Field(..., examples=["semifurnished"])
    luxury_category: str = Field(..., examples=["High"])
    floor_category: str = Field(..., examples=["Mid Floor"])


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    low_price_cr: float
    high_price_cr: float
    point_estimate_cr: float
    model_version: str


class RecommendationItem(BaseModel):
    property: str
    score: float


class NearbyItem(BaseModel):
    property: str
    distance_km: float


class ModelMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    trained_at: str
    r2_cv_mean: float
    mae: float
    git_sha: str | None = None


class PredictionOptions(BaseModel):
    property_type: list[str]
    sector: list[str]
    bedroom: list[float]
    bathroom: list[float]
    balcony: list[str]
    agePossession: list[str]
    furnishing_type: list[str]
    luxury_category: list[str]
    floor_category: list[str]
