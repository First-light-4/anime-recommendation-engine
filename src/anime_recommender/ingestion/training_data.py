"""One-pass complete-user sampling and checksum-verified local cache."""

import hashlib
import heapq
import io
import json
from pathlib import Path

import pandas as pd
import requests

from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids, require_columns


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def sample_complete_users(chunks, sample_users, seed):
    """Keep globally smallest hash priorities, with all rows of final users.

    The threshold only decreases: a rejected/evicted user cannot later qualify.
    Hence a final user is retained from their FIRST encounter, regardless of sort
    order or chunk boundaries. Storage holds only current winners' histories.
    """
    heap, histories = [], {}
    rows_seen = 0
    for chunk in chunks:
        require_columns(chunk, ["userID", "animeID", "rating"], "ratings")
        chunk = chunk[["userID", "animeID", "rating"]].copy()
        chunk["userID"] = positive_ids(chunk["userID"], "ratings.userID")
        chunk["animeID"] = positive_ids(chunk["animeID"], "ratings.animeID")
        for user_id in chunk["userID"].unique():
            user_id = int(user_id)
            if user_id in histories:
                continue
            score = int.from_bytes(hashlib.sha256(f"{seed}:{user_id}".encode()).digest(), "big")
            entry = (-score, -user_id)
            if len(heap) < sample_users:
                heapq.heappush(heap, entry)
                histories[user_id] = []
            elif entry > heap[0]:
                evicted = -heapq.heapreplace(heap, entry)[1]
                del histories[evicted]
                histories[user_id] = []
        selected = chunk.loc[chunk["userID"].isin(histories)]
        for user_id, group in selected.groupby("userID", sort=False):
            histories[int(user_id)].append(group.copy())
        rows_seen += len(chunk)
        if rows_seen % 5_000_000 < len(chunk):
            print(f"Scanned {rows_seen:,} source rows; retaining at most {sample_users} users", flush=True)
    if len(histories) < sample_users:
        raise DataValidationError(f"Requested {sample_users} users, source has only {len(histories)}")
    sample = pd.concat([part for user in sorted(histories) for part in histories[user]], ignore_index=True)
    sample = sample.sort_values(["userID", "animeID", "rating"], kind="stable").reset_index(drop=True)
    return sample, rows_seen


class HashingReader(io.RawIOBase):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.raw.read(len(buffer))
        self.digest.update(block)
        self.bytes_read += len(block)
        buffer[:len(block)] = block
        return len(block)


def acquire(config, root, allow_full_scan=False):
    cache = root / config["cache_dir"]
    source = config["source"]
    parameters = {"source": source, "sample_users": config["sample_users"], "seed": config["seed"],
                  "sampling_version": "sha256-bottom-k-complete-users-v1"}
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or "parameters" not in manifest or not isinstance(manifest.get("cache_sha256"), dict):
                raise ValueError("Invalid manifest structure")
            if set(manifest["cache_sha256"]) != {"sample_interactions.parquet", "anime_catalog.csv"}:
                raise ValueError("Invalid cached file list")
        except (ValueError, TypeError) as error:
            raise DataValidationError("Invalid cache manifest; no automatic re-download") from error
        if manifest["parameters"] != parameters:
            raise DataValidationError("Cache parameters differ; choose a new cache_dir in config")
        for name, expected_hash in manifest["cache_sha256"].items():
            path = cache / name
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise DataValidationError(f"Cached file missing or checksum mismatch: {name}")
        return pd.read_parquet(cache / "sample_interactions.parquet"), pd.read_csv(
            cache / "anime_catalog.csv", dtype="string"), manifest
    if not allow_full_scan:
        raise DataValidationError("No verified cache. A complete source scan is required; use --allow-full-scan only with approval")
    cache.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}"
    catalog_path = cache / "anime_catalog.csv"
    if not catalog_path.exists() or file_sha256(catalog_path) != source["catalog_sha256"]:
        with requests.get(base + "/animes.csv", stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            body = response.raw.read(source["catalog_bytes"] + 1)
        if len(body) != source["catalog_bytes"] or hashlib.sha256(body).hexdigest() != source["catalog_sha256"]:
            raise DataValidationError("Catalog length/checksum mismatch")
        temporary = cache / "anime_catalog.csv.tmp"
        temporary.write_bytes(body)
        temporary.replace(catalog_path)
    with requests.get(base + "/ratings.csv", stream=True, timeout=(10, 120),
                      headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        reader = HashingReader(response.raw)
        with io.BufferedReader(reader) as stream:
            chunks = pd.read_csv(stream, chunksize=config["chunk_rows"], dtype="string")
            sample, scanned = sample_complete_users(chunks, config["sample_users"], config["seed"])
        if reader.bytes_read != source["ratings_bytes"] or reader.digest.hexdigest() != source["ratings_sha256"]:
            raise DataValidationError("Ratings length/checksum mismatch: sample is not certified complete")
    atomic_parquet(sample, cache / "sample_interactions.parquet")
    manifest = {"parameters": parameters, "scanned_source_rows": scanned,
                "source_ratings_bytes": reader.bytes_read, "source_ratings_sha256": reader.digest.hexdigest(),
                "selected_rows": len(sample), "selected_users": int(sample["userID"].nunique()),
                "cache_sha256": {name: file_sha256(cache / name) for name in ["sample_interactions.parquet", "anime_catalog.csv"]}}
    atomic_json(manifest_path, manifest)
    return sample, pd.read_csv(catalog_path, dtype="string"), manifest
