"""Descriptive quality checks, not recommendation-model evaluation."""

import pandas as pd

from anime_recommender.preprocessing.user_data import STATUS_NAMES, nested, raw_user_frame, status_distribution


def profile_summary(profile, nickname):
    if profile is None:
        return pd.DataFrame(columns=["user_id", "nickname", "exact_nickname_match"])
    return pd.DataFrame([{"user_id": profile["id"], "nickname": profile.get("nickname"),
                          "exact_nickname_match": profile.get("nickname") == nickname}])


def rating_distribution(frame):
    return frame.loc[frame["user_rating"].gt(0).fillna(False), "user_rating"].value_counts().sort_index().rename_axis("User rating").to_frame("Count")


def rating_summary(frame):
    rated = frame.loc[frame["user_rating"].gt(0).fillna(False), "user_rating"]
    mal = frame["myanimelist_id"].gt(0).fillna(False)
    return pd.Series({"rated_count": len(rated), "rating_min": rated.min() if len(rated) else pd.NA,
        "rating_max": rated.max() if len(rated) else pd.NA,
        "zero_ratings": int(frame["user_rating"].eq(0).sum()),
        "missing_ratings": int(frame["user_rating"].isna().sum()),
        "favorites": int(frame["is_fav"].sum()), "mal_id_count": int(mal.sum()),
        "mal_id_share": float(mal.mean()) if len(frame) else pd.NA}, name="metric")


def user_quality_report(rows, frame):
    raw = raw_user_frame(rows)
    conversions = {col: int((raw[col].notna() & frame[col].isna()).sum())
                   for col in ["anime_id", "list_id", "date", "myanimelist_id", "user_rating", "is_fav"]}
    mal = frame["myanimelist_id"].gt(0).fillna(False)
    quality = pd.Series({"records": len(frame), "unique_anime_ids": frame["anime_id"].nunique(),
        "duplicate_anime_ids": int(frame["anime_id"].dropna().duplicated().sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "missing_anime_ids": int(frame["anime_id"].isna().sum()),
        "invalid_nonpositive_anime_ids": int(frame["anime_id"].le(0).sum()),
        "missing_watch_status": int(frame["list_id"].isna().sum()),
        "missing_status_and_favorite": int((frame["list_id"].isna() & frame["is_fav"].fillna(False)).sum()),
        "unknown_watch_status": int((frame["list_id"].notna() & ~frame["list_id"].isin(STATUS_NAMES)).sum()),
        "missing_ratings": int(frame["user_rating"].isna().sum()),
        "negative_ratings": int(frame["user_rating"].lt(0).sum()),
        "invalid_favorite_values": sum(not isinstance(v, bool) for v in raw["is_fav"]),
        "valid_mal_ids": int(mal.sum()), "mal_id_share": float(mal.mean()) if len(frame) else pd.NA,
        "duplicate_mal_ids": int(frame.loc[mal, "myanimelist_id"].duplicated().sum())}, name="Value")
    missing_status = [row for row in rows if nested(row, "user", "list", "list", "id") is None]
    shapes = pd.DataFrame([{"user_type": type(row.get("user")).__name__,
        "user_list_type": type(nested(row, "user", "list")).__name__,
        "nested_list_type": type(nested(row, "user", "list", "list")).__name__,
        "is_fav": nested(row, "user", "list", "is_fav")} for row in missing_status],
        columns=["user_type", "user_list_type", "nested_list_type", "is_fav"])
    return {"summary": quality, "coercions": pd.Series(conversions, name="Coerced to NA"),
            "columns": pd.DataFrame({"dtype": frame.dtypes.astype(str), "missing": frame.isna().sum()}),
            "missing_status_shapes": shapes}


def compare_favorites(frame, favorite_rows):
    expected = set(frame.loc[frame["is_fav"].fillna(False), "anime_id"].dropna())
    invalid_expected = int((frame["is_fav"].fillna(False) &
                            (frame["anime_id"].isna() | frame["anime_id"].le(0).fillna(False))).sum())
    valid = [row.get("anime_id") for row in favorite_rows if type(row.get("anime_id")) is int and row["anime_id"] > 0]
    invalid = len(favorite_rows) - len(valid)
    actual = set(valid)
    return {"favorite_records": len(favorite_rows), "unique_favorite_ids": len(actual),
        "invalid_favorite_ids": invalid, "invalid_list_favorite_ids": invalid_expected,
        "duplicate_favorite_ids": len(valid) - len(actual),
        "favorite_membership_matches": expected == actual and invalid == 0 and invalid_expected == 0}


def compare_rating_statistics(frame, statistics):
    stats = pd.DataFrame(statistics, columns=["rating", "count"])
    stats = stats.apply(pd.to_numeric, errors="coerce")
    valid = (stats.notna().all().all() and stats["rating"].between(0, 10).all()
             and stats["count"].ge(0).all() and stats["count"].mod(1).eq(0).all())
    if not valid:
        return {"matches": None, "comparison": pd.DataFrame(), "stats_zero_bucket": None}
    server = stats.loc[stats["rating"].gt(0)].groupby("rating")["count"].sum()
    local = rating_distribution(frame)["Count"]
    comparison = pd.concat([local.rename("List"), server.rename("Profile stats")], axis=1).fillna(0)
    return {"matches": bool(comparison["List"].eq(comparison["Profile stats"]).all()),
            "comparison": comparison, "stats_zero_bucket": int(stats.loc[stats["rating"].eq(0), "count"].sum())}


def website_comparison(frame, website_counts=None, complete=True):
    labels = ["Completed", "Planned", "Watching", "Dropped", "On hold", "Favorites"]
    counts = status_distribution(frame)["API"].to_dict()
    counts["Favorites"] = int(frame["is_fav"].sum())
    supplied = website_counts or {}
    result = pd.DataFrame({"API": {key: counts.get(key, 0) for key in labels},
        "Website": pd.Series({key: supplied.get(key) for key in labels}, dtype="Int64")})
    result["Match"] = result["API"].astype("Int64").eq(result["Website"])
    if not complete:
        result["Match"] = pd.NA
    return result
