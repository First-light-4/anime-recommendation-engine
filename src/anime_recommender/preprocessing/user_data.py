"""Normalize list snapshots without inventing watch statuses or rating semantics."""

import pandas as pd


STATUS_NAMES = {0: "Watching", 1: "Planned", 2: "Completed", 3: "Dropped", 5: "On hold"}
FIELD_PATHS = {
    "anime_id": ("anime_id",), "title": ("title",),
    "user_rating": ("user", "rating"), "list_id": ("user", "list", "list", "id"),
    "status_alias": ("user", "list", "list", "href"),
    "is_fav": ("user", "list", "is_fav"), "date": ("date",),
    "myanimelist_id": ("remote_ids", "myanimelist_id"),
}


def nested(row: dict, *keys):
    value = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def raw_user_frame(rows: list[dict]) -> pd.DataFrame:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("User lists must be a list of objects")
    return pd.DataFrame([{name: nested(row, *path) for name, path in FIELD_PATHS.items()}
                         for row in rows], columns=list(FIELD_PATHS))


def normalize_user_lists(rows: list[dict]) -> pd.DataFrame:
    """Keep duplicates and zeros. Invalid/fractional integer fields become NA.

    Raw objects are unchanged; user_quality_report reports coercions separately.
    """
    frame = raw_user_frame(rows)
    for col in ["anime_id", "list_id", "date", "myanimelist_id"]:
        values = pd.to_numeric(frame[col], errors="coerce")
        valid = values.notna() & values.between(-(2**63), 2**63 - 1) & values.mod(1).eq(0)
        valid &= ~frame[col].map(lambda value: isinstance(value, bool))
        frame[col] = values.where(valid).astype("Int64")
    frame["user_rating"] = pd.to_numeric(frame["user_rating"], errors="coerce").astype("Float64")
    frame["is_fav"] = frame["is_fav"].map(
        lambda value: value if isinstance(value, bool) else pd.NA).astype("boolean")
    return frame


def status_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["list_id"].map(STATUS_NAMES).astype("string")
    unknown = frame["list_id"].notna() & labels.isna()
    labels.loc[unknown] = frame.loc[unknown, "list_id"].map(lambda value: f"Other ({value})")
    counts = labels.fillna("No status").value_counts()
    order = list(STATUS_NAMES.values()) + [name for name in counts.index if name not in STATUS_NAMES.values()]
    return counts.reindex(order, fill_value=0).rename_axis("Status").to_frame("API")
