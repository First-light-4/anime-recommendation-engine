"""Shared records and invariants. No pipeline, network or stage imports."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TypedDict
import numpy as np
import pandas as pd

ItemMapping: TypeAlias = pd.DataFrame
Fingerprint: TypeAlias = str

class ModelMetadata(TypedDict, total=False):
    format_version: int
    model_version: str
    training_config: dict
    dataset_config: dict
    train_fingerprint: Fingerprint
    mapping_fingerprint: Fingerprint
    files_sha256: dict[str, str]

@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    version: str
    model_type: str = "als"
    train_fingerprint: Fingerprint | None = None
    mapping_fingerprint: Fingerprint | None = None

class DataValidationError(ValueError):
    """A critical source or configuration contract was violated."""

def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise DataValidationError(f"{label}: missing columns: {sorted(missing)}")

def positive_ids(values, label):
    # IDs are decimal positive integers, not booleans, fractions, or infinity.
    valid = values.astype("string").str.fullmatch(r"[1-9][0-9]*").fillna(False)
    if not valid.all():
        raise DataValidationError(f"{label}: {int((~valid).sum())} invalid/missing IDs")
    try:
        return values.astype("int64")
    except (ValueError, OverflowError) as error:
        raise DataValidationError(f"{label}: IDs exceed int64 range") from error

@dataclass
class RecommendationResult:
    recommendations: pd.DataFrame
    available: bool
    diagnostics: dict
    reason: str | None = None

def validate_item_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    require_columns(mapping, ["dataset_anime_id", "myanimelist_id", "matrix_col"], "model item mapping")
    frame = mapping[["dataset_anime_id", "myanimelist_id", "matrix_col"]].copy()
    for col in ("dataset_anime_id", "myanimelist_id"):
        frame[col] = positive_ids(frame[col], col)
        if frame[col].duplicated().any():
            raise DataValidationError(f"Model mapping is ambiguous: {col}")
    if frame["matrix_col"].tolist() != list(range(len(frame))):
        raise DataValidationError("Model item columns must be contiguous and ordered")
    return frame.reset_index(drop=True)

class InferenceModel(Protocol):
    def recommend(self, user_context, top_k=10, exclude_seen=True, seen_items=None) -> RecommendationResult: ...

@dataclass
class ModelBundle:
    als: Any
    popularity: pd.DataFrame
    items: ItemMapping
    train_users: pd.DataFrame
    metadata: ModelMetadata

@dataclass(repr=False)
class ModelWorkspace:
    bundle: object
    inputs: object
    catalog: pd.DataFrame
    franchise_config: dict | None = None
    franchise_mapping: object | None = None

def validate_factors(model, users, items):
    if (model.user_factors.shape != (users, model.factors)
            or model.item_factors.shape != (items, model.factors)
            or not np.isfinite(model.user_factors).all() or not np.isfinite(model.item_factors).all()):
        raise DataValidationError("ALS factor orientation, dimensions or finite values are invalid")
