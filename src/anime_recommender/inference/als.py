"""Score all train items from a fresh fold-in factor; scores are NOT probabilities."""

from anime_recommender.inference.fold_in import fold_in
from anime_recommender.inference.recommendations import (
    RecommendationResult, context_ids, rank_scores, unavailable_result, validate_item_mapping,
)


class ALSRecommender:
    model_name = "als"

    def __init__(self, model, item_mapping):
        self.model = model
        self.item_mapping = validate_item_mapping(item_mapping)

    def recommend(self, user_context=(), top_k=10, exclude_seen=True, seen_items=None):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        ids = context_ids(user_context)
        folded = fold_in(self.model, ids, self.item_mapping)
        if not folded.available:
            return unavailable_result(folded.diagnostics)
        seen = ids if seen_items is None else sorted(set(ids) | set(context_ids(seen_items)))
        scores = self.model.item_factors @ folded.factor
        frame = rank_scores(scores, self.item_mapping, self.model_name, seen, top_k, exclude_seen)
        return RecommendationResult(frame, True, folded.diagnostics)
