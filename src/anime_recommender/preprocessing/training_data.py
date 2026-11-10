"""Validation and conservative cleaning of training tables; no ALS transforms."""

import pandas as pd


class DataValidationError(ValueError):
    """A critical source or configuration contract was violated."""


def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise DataValidationError(f"{label}: missing columns: {sorted(missing)}")


def positive_ids(values, label):
    # IDs are decimal positive integers, not booleans, fractions, or infinity.
    valid = values.astype("string").str.fullmatch(r"[1-9][0-9]*").fillna(False)
    if not valid.all():
        raise DataValidationError(f"{label}: {int((~valid).sum())} invalid/missing IDs")
    try:
        return values.astype("int64")
    except (ValueError, OverflowError) as error:
        raise DataValidationError(f"{label}: IDs exceed int64 range") from error


def prepare_catalog(raw):
    require_columns(raw, ["animeID", "title", "mal_url"], "catalog")
    catalog = raw.copy()
    catalog["animeID"] = positive_ids(catalog["animeID"], "catalog.animeID")
    if catalog["animeID"].duplicated().any():
        raise DataValidationError("catalog: duplicate primary animeID")
    if catalog["title"].isna().any() or catalog["title"].astype("string").str.strip().eq("").any():
        raise DataValidationError("catalog: missing/empty required title")
    mal = catalog["mal_url"].astype("string").str.extract(
        r"^https?://(?:www\.)?myanimelist\.net/anime/([1-9][0-9]*)(?:/[^?#]*)?(?:[?#].*)?$",
        expand=False,
    )
    # Safe int64 conversion with an explicit domain check.
    parsed = pd.to_numeric(mal, errors="coerce")
    invalid = parsed.isna() | parsed.gt(2**63 - 1)
    catalog["myanimelist_id"] = parsed.mask(invalid).astype("Int64")
    duplicate = catalog["myanimelist_id"].notna() & catalog["myanimelist_id"].duplicated(keep=False)
    reason = pd.Series("", index=catalog.index, dtype="string")
    reason.loc[invalid] = "missing_or_invalid_mal_url"
    reason.loc[duplicate] = "ambiguous_mal_id"
    quarantine = catalog.loc[reason.ne("")].copy()
    quarantine["exclusion_reason"] = reason.loc[reason.ne("")]
    clean = catalog.loc[reason.eq("")].rename(columns={"animeID": "dataset_anime_id"})
    mapping = clean[["dataset_anime_id", "myanimelist_id"]].copy()
    report = {
        "input_rows": len(catalog), "output_rows": len(clean),
        "missing_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "missing_mal_urls": int(raw["mal_url"].isna().sum()),
        "invalid_or_missing_mal_urls": int(invalid.sum()),
        "ambiguous_mal_rows": int(duplicate.sum()),
        "mapping_success_share": len(mapping) / len(catalog) if len(catalog) else None,
    }
    return clean, mapping, quarantine, report


def clean_interactions(raw, mapping, rating_min=0, rating_max=10):
    require_columns(raw, ["userID", "animeID", "rating"], "ratings")
    frame = raw.copy()
    frame["userID"] = positive_ids(frame["userID"], "ratings.userID")
    frame["animeID"] = positive_ids(frame["animeID"], "ratings.animeID")
    frame["rating_raw"] = frame["rating"].astype("string")
    numeric = pd.to_numeric(frame["rating"], errors="coerce").astype("float64")
    missing = frame["rating"].isna() | frame["rating_raw"].str.strip().eq("").fillna(False)
    invalid = ~missing & ~numeric.between(rating_min, rating_max)
    frame["rating"] = numeric.astype("float64")
    removals = {"missing_rating": int(missing.sum()), "invalid_rating": int(invalid.sum())}
    frame = frame.loc[~missing & ~invalid].copy()
    exact = frame.duplicated(["userID", "animeID", "rating_raw"])
    removals["exact_duplicate"] = int(exact.sum())
    frame = frame.loc[~exact].copy()
    pair_size = frame.groupby(["userID", "animeID"])["rating"].transform("size")
    conflicts = frame.groupby(["userID", "animeID"])["rating"].transform("nunique").gt(1)
    conflict_pairs = frame.loc[conflicts, ["userID", "animeID"]].drop_duplicates()
    repeated_pairs = frame.loc[pair_size.gt(1), ["userID", "animeID"]].drop_duplicates()
    removals["conflicting_pair_rows"] = int(conflicts.sum())
    frame = frame.loc[~conflicts].copy()
    repeated = frame.duplicated(["userID", "animeID"])
    removals["equivalent_rating_duplicate"] = int(repeated.sum())
    frame = frame.loc[~repeated].rename(columns={"userID": "dataset_user_id", "animeID": "dataset_anime_id"})
    mapped = frame["dataset_anime_id"].isin(mapping["dataset_anime_id"])
    removals["unmapped_anime"] = int((~mapped).sum())
    result = frame.loc[mapped, ["dataset_user_id", "dataset_anime_id", "rating", "rating_raw"]].merge(
        mapping, on="dataset_anime_id", how="left", validate="many_to_one")
    result = result.sort_values(["dataset_user_id", "dataset_anime_id"]).reset_index(drop=True)
    report = {
        "input_rows": len(raw), "output_rows": len(result),
        "raw_repeated_pairs": int(raw.groupby(["userID", "animeID"]).size().gt(1).sum()),
        "input_users": int(raw["userID"].nunique()), "input_anime": int(raw["animeID"].nunique()),
        "output_users": int(result["dataset_user_id"].nunique()),
        "output_anime": int(result["dataset_anime_id"].nunique()),
        "repeated_pairs_after_exact_dedup": len(repeated_pairs),
        "conflicting_pairs": len(conflict_pairs), "removed_by_reason": removals,
        "missing_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "mapping_success_share_before_mapping_filter": float(mapped.mean()) if len(mapped) else None,
        "zero_ratings_preserved": int(result["rating"].eq(0).sum()),
    }
    if len(raw) != len(result) + sum(removals.values()):
        raise DataValidationError("Removal accounting mismatch: input != output + removed rows")
    return result, report
