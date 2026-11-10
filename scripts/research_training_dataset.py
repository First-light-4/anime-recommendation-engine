"""ARE-003: bounded, read-only dataset EDA and optional private-profile ID check.

Run with .venv Python from any directory. Requires the research extra.
No raw data or profile history is written. Sample is deterministic byte windows,
NOT a random sample or complete user histories. No training is performed.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv


REVISION = "6d0f78003440a90ca9a417389f13c47387f0c942"
ROOT_URL = f"https://huggingface.co/datasets/mramazan/User-Animelist-Dataset/resolve/{REVISION}"
RATINGS_BYTES = 2_254_824_347
CATALOG_BYTES = 10_816_239
WINDOW_BYTES = 262_144
WINDOW_STARTS = [0, RATINGS_BYTES // 4, RATINGS_BYTES // 2,
                 3 * RATINGS_BYTES // 4, RATINGS_BYTES - WINDOW_BYTES]


def download_bounded(url, max_bytes, start=None):
    headers = {"Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + max_bytes - 1}"
    with requests.get(url, headers=headers, stream=True, timeout=(10, 60)) as response:
        expected = 206 if start is not None else 200
        if response.status_code != expected:
            raise RuntimeError(f"Dataset HTTP {response.status_code}; stopped")
        if start is not None:
            expected_range = f"bytes {start}-{start + max_bytes - 1}/{RATINGS_BYTES}"
            if response.headers.get("Content-Range") != expected_range:
                raise RuntimeError("Unexpected Content-Range; stopped")
        data = response.raw.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError("Download exceeded byte limit; stopped")
        if len(data) != max_bytes:
            raise RuntimeError("Unexpected download length; stopped")
        return data


def parse_window(data, start):
    # Ratings CSV has three numeric columns, no quoted multiline strings.
    lines = data.split(b"\n")
    # Always discard the final possibly incomplete row. At nonzero offsets,
    # discard the first possibly incomplete row too.
    complete = lines[1:-1]
    if start == 0 and lines[0].strip() != b"userID,animeID,rating":
        raise ValueError("Unexpected ratings header")
    return pd.read_csv(io.BytesIO(b"\n".join(complete)),
                       names=["userID", "animeID", "rating"])


def catalog_mapping(catalog):
    mapping = catalog[["animeID", "mal_url"]].copy()
    mapping["mal_id"] = pd.to_numeric(
        mapping["mal_url"].str.extract(
            r"^https?://(?:www\.)?myanimelist\.net/anime/(\d+)(?:/[^?#]*)?(?:[?#].*)?$",
            expand=False), errors="coerce").astype("Int64")
    # Exclude ALL ambiguous mappings; never silently select an arbitrary row.
    ambiguous = mapping["animeID"].duplicated(keep=False) | mapping["mal_id"].duplicated(keep=False)
    safe = mapping.loc[~ambiguous & mapping["mal_id"].gt(0).fillna(False)]
    return mapping, safe


def describe_counts(counts):
    return {str(k): float(v) for k, v in counts.describe(
        percentiles=[.25, .5, .75, .95, .99]).items()}


def analyze(catalog, ratings):
    mapping, safe = catalog_mapping(catalog)
    pairs = ratings[["userID", "animeID"]].drop_duplicates()
    users, items = pairs["userID"].nunique(), pairs["animeID"].nunique()
    return {
        "revision": REVISION,
        "sampling": "five fixed byte windows; partial user histories; NOT representative",
        "sample_rows": len(ratings), "sample_users": users, "sample_items": items,
        "sample_density": len(pairs) / (users * items) if users and items else None,
        "sample_sparsity": 1 - len(pairs) / (users * items) if users and items else None,
        "rating_distribution": {str(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        "rating_summary": describe_counts(ratings["rating"]),
        "rating_dtype": str(ratings["rating"].dtype),
        "sample_interactions_per_user": describe_counts(ratings.groupby("userID").size()),
        "sample_interactions_per_item": describe_counts(ratings.groupby("animeID").size()),
        "missing": {k: int(v) for k, v in ratings.isna().sum().items()},
        "duplicate_rows": int(ratings.duplicated().sum()),
        "duplicate_pairs": int(ratings.duplicated(["userID", "animeID"]).sum()),
        "conflicting_pairs": int(ratings.groupby(["userID", "animeID"])["rating"].nunique().gt(1).sum()),
        "sample_items_missing_catalog": len(set(ratings["animeID"]) - set(catalog["animeID"])),
        "catalog_rows": len(catalog), "catalog_unique_items": catalog["animeID"].nunique(),
        "catalog_missing": {k: int(v) for k, v in catalog.isna().sum().items()},
        "catalog_duplicate_ids": int(mapping["animeID"].duplicated().sum()),
        "catalog_invalid_mal_urls": int(mapping["mal_id"].isna().sum()),
        "catalog_duplicate_mal_ids": int(mapping["mal_id"].dropna().duplicated().sum()),
        "safe_mapping_rows": len(safe),
        "raw_anime_id_equals_mal_id": int(mapping["animeID"].eq(mapping["mal_id"]).sum()),
        "positive_at_least_7": int(ratings["rating"].ge(7).sum()),
    }


def profile_coverage(nickname, safe, ratings):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    token = os.getenv("YUMMYANIME_APPLICATION_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Public application token missing")

    def get(path, label):
        try:
            response = requests.get("https://api.yani.tv" + path,
                headers={"X-Application": token, "Accept": "application/json"},
                timeout=(10, 30), allow_redirects=False)
        except requests.RequestException:
            raise RuntimeError("Profile network failure; stopped") from None
        print(f"{label}: HTTP {response.status_code}")
        if response.status_code != 200:
            raise RuntimeError("Profile API failure; stopped")
        payload = response.json()
        if not isinstance(payload, dict) or "response" not in payload or payload.get("error") or payload.get("errors"):
            raise RuntimeError("Unexpected API envelope; stopped")
        return payload["response"]

    profile = get("/users/" + quote(nickname, safe=""), "profile")
    user_id = profile.get("id") if isinstance(profile, dict) else None
    if type(user_id) is not int:
        raise RuntimeError("Profile ID missing; stopped")
    rows = get(f"/users/{user_id}/lists", "lists")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("Unexpected profile list; stopped")
    mal_ids = [(row.get("remote_ids") or {}).get("myanimelist_id") for row in rows]
    valid = [v for v in mal_ids if type(v) is int and v > 0]
    ids = set(valid)
    known = set(safe["mal_id"].dropna())
    sampled_known = set(safe.loc[safe["animeID"].isin(ratings["animeID"]), "mal_id"])
    return {"profile_records": len(rows), "valid_mal_records": len(valid),
        "unique_profile_mal_ids": len(ids), "duplicate_profile_mal_ids": len(valid) - len(ids),
        "catalog_matches": len(ids & known), "catalog_unknown": len(ids - known),
        "catalog_coverage": len(ids & known) / len(ids) if ids else None,
        "sample_item_matches": len(ids & sampled_known),
        "sample_item_coverage": len(ids & sampled_known) / len(ids) if ids else None,
        "note": "One profile only. Catalog presence != post-filter training support. No IDs saved."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-nickname", help="Optional one owned public profile; never saved")
    args = parser.parse_args()
    catalog_bytes = download_bounded(ROOT_URL + "/animes.csv", CATALOG_BYTES)
    if hashlib.sha256(catalog_bytes).hexdigest() != "5bec78930e6e7633156046338947954c7e619d9615116bec13d9fc4e04221175":
        raise RuntimeError("Catalog checksum mismatch; stopped")
    catalog = pd.read_csv(io.BytesIO(catalog_bytes))
    samples, manifest = [], []
    for start in WINDOW_STARTS:
        data = download_bounded(ROOT_URL + f"/ratings.csv?window={start}", WINDOW_BYTES, start)
        samples.append(parse_window(data, start))
        manifest.append({"start": start, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    ratings = pd.concat(samples, ignore_index=True)
    result = analyze(catalog, ratings)
    result["catalog_sha256"] = hashlib.sha256(catalog_bytes).hexdigest()
    result["windows"] = manifest
    print(json.dumps(result, indent=2))
    if args.profile_nickname:
        _, safe = catalog_mapping(catalog)
        print(json.dumps(profile_coverage(args.profile_nickname, safe, ratings), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, ValueError, RuntimeError, KeyError) as error:
        # Library error strings can expose private URLs; do not print them.
        print(f"Stopped: {type(error).__name__}; no raw data saved.")
        raise SystemExit(1) from None
