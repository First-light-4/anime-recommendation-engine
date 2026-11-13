"""Shared Top-N contract, deterministic ranking and known-item exclusion."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids, require_columns


OUTPUT_COLUMNS = ["dataset_anime_id", "myanimelist_id", "score", "rank", "model_name"]


@dataclass
class RecommendationResult:
    recommendations: pd.DataFrame
    available: bool
    diagnostics: dict
    reason: str | None = None


def context_ids(values) -> list[int]:
    """Binary contexts are deduplicated. Unknown positive IDs remain cold."""
    result = list(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0 for v in result):
        raise ValueError("Context/seen IDs must be positive integers in dataset ID space")
    return sorted(set(int(v) for v in result))


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


def rank_scores(scores, mapping, model_name, seen_ids=(), top_k=10, exclude_seen=True):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    scores = np.asarray(scores)
    if scores.shape != (len(mapping),) or not np.isfinite(scores).all():
        raise DataValidationError("Scores must be a finite vector aligned with the model mapping")
    ranked = mapping[["dataset_anime_id", "myanimelist_id"]].copy()
    ranked["score"] = scores.astype("float64")
    if exclude_seen:
        ranked = ranked.loc[~ranked["dataset_anime_id"].isin(context_ids(seen_ids))]
    ranked = ranked.sort_values(["score", "dataset_anime_id"], ascending=[False, True], kind="stable").head(top_k)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["model_name"] = model_name
    return ranked[OUTPUT_COLUMNS].reset_index(drop=True)


def unavailable_result(diagnostics):
    return RecommendationResult(pd.DataFrame(columns=OUTPUT_COLUMNS), False, diagnostics,
                                "no_warm_positive_context")
