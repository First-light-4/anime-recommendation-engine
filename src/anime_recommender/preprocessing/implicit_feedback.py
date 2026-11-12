"""Explicit ratings to positive feedback; never mutate source ratings."""

import pandas as pd

from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids, require_columns


def positive_feedback(interactions: pd.DataFrame, threshold: int = 5,
                      confidence: float = 1.0) -> pd.DataFrame:
    """Return positive rows with binary preference and uniform confidence.

    Missing ratings remain unobserved, never dislikes. No learned weights yet.
    """
    if type(threshold) is not int or threshold not in (5, 6, 7, 8):
        raise DataValidationError("positive_threshold must be 5, 6, 7 or 8")
    if confidence != 1.0 or isinstance(confidence, bool):
        raise DataValidationError("Only binary feedback with uniform confidence=1 is supported")
    require_columns(interactions, ["dataset_user_id", "dataset_anime_id", "rating"], "clean interactions")
    canonical = interactions.copy()
    for col in ["dataset_user_id", "dataset_anime_id"]:
        canonical[col] = positive_ids(interactions[col], col)
    if canonical.duplicated(["dataset_user_id", "dataset_anime_id"]).any():
        raise DataValidationError("Input contains duplicate user-item pairs; run ARE-004 cleaning first")
    numeric = pd.to_numeric(interactions["rating"], errors="coerce")
    invalid = interactions["rating"].notna() & (numeric.isna() | ~numeric.between(0, 10))
    if invalid.any():
        raise DataValidationError("Input has invalid ratings outside numeric [0,10]")
    result = canonical.loc[numeric.ge(threshold).fillna(False)].copy()
    result["positive"] = 1
    result["confidence"] = pd.Series(1.0, index=result.index, dtype="float32")
    return result.sort_values(["dataset_user_id", "dataset_anime_id"]).reset_index(drop=True)


def threshold_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"threshold": value, "positive_interactions": len(frame),
        "users": frame["dataset_user_id"].nunique(), "anime": frame["dataset_anime_id"].nunique()}
        for value in (5, 6, 7, 8) for frame in [positive_feedback(interactions, value)]])
