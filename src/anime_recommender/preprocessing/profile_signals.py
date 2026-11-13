"""Binary experimental YummyAnime feedback; no HTTP, persistence or model fitting."""

from dataclasses import dataclass

import pandas as pd

from anime_recommender.evaluation.training_data import map_user_to_training
from anime_recommender.preprocessing.user_data import normalize_user_lists, raw_user_frame


@dataclass(repr=False)
class ProfileSignals:
    frame: pd.DataFrame
    positive_ids: list[int]
    seen_ids: list[int]
    coverage: pd.Series


def normalize_profile_history(rows, favorites):
    """List rows retain their statuses; favorites endpoint adds membership only.

    Favorite-only records are included so both positive and seen sets are complete.
    Raw dictionaries and normalized source fields are not modified.
    """
    frame = normalize_user_lists(rows)
    fav = normalize_user_lists(favorites)
    frame["rating_missing_in_api"] = raw_user_frame(rows).user_rating.isna()
    fav["rating_missing_in_api"] = raw_user_frame(favorites).user_rating.isna()
    favorite_ids = set(fav.anime_id.dropna())
    extra = fav.loc[~fav.anime_id.isin(frame.anime_id)].copy()
    if not extra.empty:
        frame = extra.reset_index(drop=True) if frame.empty else pd.concat([frame, extra], ignore_index=True)
    frame.loc[frame.anime_id.isin(favorite_ids), "is_fav"] = True
    return frame


def prepare_profile_signals(frame, catalog_mapping, train_items):
    """Map via MAL, with a planned/dropped veto and unit positive confidence.

    Completed (list_id=2) with rating=0/NA, rating in [5,10], or favorite is positive.
    Invalid ratings are not inferred to be unrated. No rating/status is rewritten.
    All mapped history, including nonpositive history, enters the seen set.
    """
    source = frame.copy()
    source["myanimelist_id"] = source.myanimelist_id.where(source.myanimelist_id.gt(0))
    joined, coverage = map_user_to_training(source, catalog_mapping, train_items)
    rating = joined.user_rating
    joined["positive_rating"] = rating.between(5, 10).fillna(False)
    missing = joined.get("rating_missing_in_api", rating.isna()).fillna(False)
    joined["completed_unrated"] = joined.list_id.eq(2).fillna(False) & (missing | rating.eq(0).fillna(False))
    joined["favorite_signal"] = joined.is_fav.fillna(False)
    blocked = joined.list_id.isin([1, 3])
    joined["positive"] = (~blocked & (joined.positive_rating | joined.completed_unrated | joined.favorite_signal)).astype(bool)
    warm_positive = joined.positive & joined.known_in_training
    positive_ids = sorted(set(joined.loc[warm_positive, "dataset_anime_id"].dropna().astype(int)))
    seen_ids = sorted(set(joined.dataset_anime_id.dropna().astype(int)))
    coverage = coverage.copy()
    coverage["positive_records"] = int(joined.positive.sum())
    coverage["cold_or_unmapped_positive_records"] = int((joined.positive & ~joined.known_in_training).sum())
    coverage["fold_in_interactions"] = len(positive_ids)
    coverage["binary_confidence"] = 1
    coverage["unmapped_history_records"] = int(joined.dataset_anime_id.isna().sum())
    return ProfileSignals(joined, positive_ids, seen_ids, coverage)
