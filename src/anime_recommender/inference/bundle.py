"""Inference factory for the neutral bundle record, with the legacy constructor."""

from anime_recommender.contracts import ModelBundle as BundleRecord, InferenceModel
from anime_recommender.inference.als import ALSRecommender
from anime_recommender.inference.popularity import PopularityRecommender


class ModelBundle(BundleRecord):
    def recommenders(self) -> dict[str, InferenceModel]:
        return {"popularity": PopularityRecommender(self.popularity, self.items),
                "als": ALSRecommender(self.als, self.items)}
