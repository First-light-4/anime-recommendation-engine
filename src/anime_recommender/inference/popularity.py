"""Apply saved popularity counts; no fitting and no personalized factors."""

import pandas as pd

from anime_recommender.inference.recommendations import (
    RecommendationResult, context_ids, rank_scores, validate_item_mapping,
)
from anime_recommender.preprocessing.training_data import DataValidationError


class PopularityRecommender:
    model_name = "popularity"

    def __init__(self, popularity: pd.DataFrame, item_mapping: pd.DataFrame):
        self.item_mapping = validate_item_mapping(item_mapping)
        if (popularity["dataset_anime_id"].tolist() != self.item_mapping["dataset_anime_id"].tolist()
                or popularity["myanimelist_id"].tolist() != self.item_mapping["myanimelist_id"].tolist()
                or popularity["matrix_col"].tolist() != self.item_mapping["matrix_col"].tolist()
                or not popularity["unique_users"].gt(0).all()):
            raise DataValidationError("Popularity counts and model mapping disagree")
        self.scores = popularity["unique_users"].to_numpy(dtype="float64")

    def recommend(self, user_context=(), top_k=10, exclude_seen=True, seen_items=None):
        context = context_ids(user_context)
        seen = context if seen_items is None else sorted(set(context) | set(context_ids(seen_items)))
        frame = rank_scores(self.scores, self.item_mapping, self.model_name, seen, top_k, exclude_seen)
        vocabulary = set(self.item_mapping["dataset_anime_id"])
        return RecommendationResult(frame, True, {"context_unique": len(context),
            "warm_context": len(set(context) & vocabulary), "cold_context": len(set(context) - vocabulary)})
