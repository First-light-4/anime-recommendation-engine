"""Local descriptive statistics and explicit MAL-based catalog matching."""

from anime_recommender.preprocessing.catalog_mapping import map_user_to_training

import pandas as pd

from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids, require_columns


def training_statistics(interactions: pd.DataFrame) -> dict:
    require_columns(interactions, ["dataset_user_id", "dataset_anime_id", "rating"], "interactions")
    users = interactions["dataset_user_id"].nunique()
    items = interactions["dataset_anime_id"].nunique()
    pairs = interactions[["dataset_user_id", "dataset_anime_id"]].drop_duplicates()
    density = len(pairs) / (users * items) if users and items else None
    return {"summary": pd.Series({"records": len(interactions), "users": users, "anime": items,
        "duplicate_pairs": len(interactions) - len(pairs),
        "zero_ratings": int(interactions["rating"].eq(0).sum()),
        "zero_rating_share": float(interactions["rating"].eq(0).mean()) if len(interactions) else None,
        "density": density, "sparsity": 1 - density if density is not None else None}),
        "ratings": interactions["rating"].value_counts().sort_index().rename_axis("Rating").to_frame("Count"),
        "per_user": interactions.groupby("dataset_user_id").size().rename("Interactions"),
        "per_anime": interactions.groupby("dataset_anime_id").size().rename("Interactions")}
