"""Source-backed title presentation; never translates or changes ranking."""

import re

import pandas as pd


ENGLISH_FIELDS = ("title_en", "english_title", "title_english", "title")
RUSSIAN_FIELDS = ("title_ru", "russian_title", "title_russian", "yummyanime_title_ru")


def _title_column(frame, fields):
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name in fields:
        if name in frame:
            values = frame[name].map(lambda value: value.strip() if isinstance(value, str) else None).astype("string")
            result = result.fillna(values.mask(values.isin(["", "—"])))
    return result.fillna("—")


def recommendation_table(recommendations):
    """Prefer explicitly localized columns; catalog primary title is EN fallback.

    The primary catalog title can be romanized rather than a licensed English title.
    alternative_title and unlabeled other_titles are never treated as Russian.
    Popularity counts must never be labeled as ALS scores.
    """
    return pd.DataFrame({"Rank": recommendations["rank"],
        "English Title": _title_column(recommendations, ENGLISH_FIELDS),
        "Russian Title": _title_column(recommendations, RUSSIAN_FIELDS),
        "ALS Score": recommendations.score.astype(object).where(recommendations.model_name.eq("als"), "—"),
        "MAL ID": recommendations.myanimelist_id}).reset_index(drop=True)


def russian_titles_by_mal(rows, requested_ids):
    """Use only unambiguous exact MAL matches and the primary YummyAnime title.

    Cyrillic confirms a Russian-script source title, not an invented translation.
    Unknown-language aliases and multiple catalog entries per MAL are ignored.
    """
    matches = {}
    for row in rows:
        remote = row.get("remote_ids")
        mal = remote.get("myanimelist_id") if isinstance(remote, dict) else None
        if type(mal) is not int or mal not in requested_ids:
            continue
        matches.setdefault(mal, []).append(row.get("title"))
    result = {}
    for mal in requested_ids:
        values = matches.get(mal, [])
        title = values[0].strip() if len(values) == 1 and isinstance(values[0], str) else None
        result[mal] = title if title and re.search(r"[А-Яа-яЁё]", title) else None
    return result
