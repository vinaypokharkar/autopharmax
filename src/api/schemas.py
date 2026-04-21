"""Pydantic v2 request/response schemas for the AutoPharmaX API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---- Requests -----------------------------------------------------------

class SinglePredictRequest(BaseModel):
    cosmic_id: int = Field(..., description="COSMIC cell-line ID (integer).")
    drug_name: str = Field(..., description="Drug name (PubChem-resolvable or already in cache).")


class CombinationPredictRequest(BaseModel):
    cosmic_id: int
    drug_a: str
    drug_b: str


class CombinationScanRequest(BaseModel):
    cosmic_id: int
    drug_a: str
    top_k: int = Field(10, ge=1, le=50)


# ---- Response pieces ----------------------------------------------------

class FeatureContribution(BaseModel):
    feature: str
    importance: float


class SinglePredictResponse(BaseModel):
    predicted_ln_ic50: float
    predicted_ic50_uM: float
    sensitivity_class: str                       # "sensitive" | "insensitive"
    is_novel_prediction: bool                    # True = pair NOT in training set (the good case)
    actual_ln_ic50: Optional[float] = None       # if the pair happens to be in merged_single
    absolute_error: Optional[float] = None
    top_cell_features: list[FeatureContribution]
    top_drug_features: list[FeatureContribution]


class CombinationPredictResponse(BaseModel):
    synergy_score: float
    synergy_class: str                           # "synergistic" | "additive" | "antagonistic"
    drug_a_ln_ic50: float
    drug_b_ln_ic50: float
    is_novel_combination: bool


class CombinationScanRow(BaseModel):
    drug_b: str
    synergy_score: float
    synergy_class: str


class CombinationScanResponse(BaseModel):
    cosmic_id: int
    drug_a: str
    top_k: int
    results: list[CombinationScanRow]


class CellLineInfo(BaseModel):
    cosmic_id: int
    cell_line_name: Optional[str] = None
    tissue: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    single_drug_ldo_r2: Optional[float] = None
    cell_features: int
    drug_features: int
    device: str
