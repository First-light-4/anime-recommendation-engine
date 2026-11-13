"""Unique-positive-user popularity fitted from train interactions only."""

import pandas as pd

from anime_recommender.inference.recommendations import validate_item_mapping
from anime_recommender.preprocessing.training_data import DataValidationError, require_columns


def train_popularity(train_interactions: pd.DataFrame, item_mapping: pd.DataFrame) -> pd.DataFrame:
    require_columns(train_interactions, ["dataset_user_id", "dataset_anime_id", "positive"], "train positives")
    mapping = validate_item_mapping(item_mapping)
    if not train_interactions["positive"].eq(1).all():
        raise DataValidationError("Popularity training requires positive train interactions")
    if set(train_interactions["dataset_anime_id"]) != set(mapping["dataset_anime_id"]):
        raise DataValidationError("Popularity vocabulary must equal train vocabulary")
    counts = train_interactions.groupby("dataset_anime_id")["dataset_user_id"].nunique()
    result = mapping.copy()
    result["unique_users"] = result["dataset_anime_id"].map(counts).astype("int64")
    return result
