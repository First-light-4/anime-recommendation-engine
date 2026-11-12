"""Local descriptive statistics and explicit MAL-based catalog matching."""

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


def map_user_to_training(user_frame: pd.DataFrame, mapping: pd.DataFrame,
                         interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Join via MAL, never via YummyAnime anime_id or dataset internal ID.

    Do not deduplicate or drop user history. Report coverage over valid unique MAL IDs.
    """
    require_columns(user_frame, ["myanimelist_id"], "user lists")
    require_columns(mapping, ["dataset_anime_id", "myanimelist_id"], "mapping")
    require_columns(interactions, ["dataset_anime_id"], "interactions")
    safe = mapping[["dataset_anime_id", "myanimelist_id"]].copy()
    for col in safe:
        safe[col] = positive_ids(safe[col], f"mapping.{col}")
        if safe[col].duplicated().any():
            raise DataValidationError(f"Ambiguous mapping: duplicate {col}")
    joined = user_frame.merge(safe, on="myanimelist_id", how="left", validate="many_to_one")
    joined["dataset_anime_id"] = joined["dataset_anime_id"].astype("Int64")
    joined["known_in_training"] = joined["dataset_anime_id"].isin(interactions["dataset_anime_id"])
    valid = user_frame["myanimelist_id"].gt(0).fillna(False)
    ids = set(user_frame.loc[valid, "myanimelist_id"])
    catalog_ids = set(safe["myanimelist_id"])
    active_ids = set(safe.loc[safe["dataset_anime_id"].isin(interactions["dataset_anime_id"]), "myanimelist_id"])
    summary = pd.Series({"profile_records": len(user_frame), "valid_mal_records": int(valid.sum()),
        "unique_mal_ids": len(ids), "duplicate_mal_ids": int(user_frame.loc[valid, "myanimelist_id"].duplicated().sum()),
        "catalog_matches": len(ids & catalog_ids), "catalog_unknown": len(ids - catalog_ids),
        "catalog_coverage": len(ids & catalog_ids) / len(ids) if ids else None,
        "training_matches": len(ids & active_ids), "training_unknown": len(ids - active_ids),
        "training_coverage": len(ids & active_ids) / len(ids) if ids else None})
    return joined, summary
