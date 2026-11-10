"""Synthetic source fixtures; tests never download a real dataset."""

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("requests")

from anime_recommender.ingestion.pipeline import run_pipeline, validate_config
from anime_recommender.ingestion.training_data import sample_complete_users
from anime_recommender.preprocessing.training_data import (
    DataValidationError, clean_interactions, prepare_catalog,
)


def mapping():
    return pd.DataFrame({"dataset_anime_id": [1, 2, 3], "myanimelist_id": [431, 1, 100]})


def ratings(rows):
    return pd.DataFrame(rows, columns=["userID", "animeID", "rating"]).astype("string")


def test_missing_columns_and_invalid_ids_fail():
    with pytest.raises(DataValidationError, match="missing columns"):
        clean_interactions(pd.DataFrame({"userID": [1]}), mapping())
    for bad in [None, "-1", "0", "1.5", "true", "inf", str(2**64)]:
        with pytest.raises(DataValidationError, match="ID"):
            clean_interactions(ratings([[bad, 1, 8]]), mapping())


def test_exact_duplicates_conflicts_and_zero_ratings():
    raw = ratings([[1, 1, 8], [1, 1, 8], [1, 2, 7], [1, 2, 9], [2, 3, 0]])
    clean, report = clean_interactions(raw, mapping())
    assert clean[["dataset_user_id", "dataset_anime_id", "rating"]].values.tolist() == [[1, 1, 8], [2, 3, 0]]
    assert clean["rating_raw"].tolist() == ["8", "0"]
    assert report["conflicting_pairs"] == 1
    assert report["removed_by_reason"]["exact_duplicate"] == 1
    assert report["removed_by_reason"]["conflicting_pair_rows"] == 2
    assert report["zero_ratings_preserved"] == 1


def test_missing_invalid_ratings_and_unmapped_rows_are_accounted():
    raw = ratings([[1, 1, None], [1, 2, "bad"], [1, 3, "inf"],
                   [2, 1, -1], [2, 2, 11], [2, 99, 8], [3, 1, "8.5"]])
    clean, report = clean_interactions(raw, mapping())
    assert clean["rating"].tolist() == [8.5]
    assert report["removed_by_reason"]["missing_rating"] == 1
    assert report["removed_by_reason"]["invalid_rating"] == 4
    assert report["removed_by_reason"]["unmapped_anime"] == 1
    assert sum(report["removed_by_reason"].values()) == len(raw) - len(clean)


def test_same_numeric_rating_with_different_text_is_not_conflict():
    clean, report = clean_interactions(ratings([[1, 1, "8"], [1, 1, "8.0"]]), mapping())
    assert len(clean) == 1
    assert report["conflicting_pairs"] == 0
    assert report["removed_by_reason"]["equivalent_rating_duplicate"] == 1


def test_mal_urls_and_all_ambiguous_mappings_quarantined():
    raw = pd.DataFrame({"animeID": [1, 2, 3, 4, 5, 6], "title": ["synthetic"] * 6,
        "mal_url": ["https://myanimelist.net/anime/431/title", "https://myanimelist.net/anime/1",
                    "https://myanimelist.net/anime/1", None, "https://evil.invalid/anime/12",
                    "https://myanimelist.net/anime/0"]})
    catalog, ids, quarantine, report = prepare_catalog(raw)
    assert ids.to_dict("records") == [{"dataset_anime_id": 1, "myanimelist_id": 431}]
    assert len(quarantine) == 5
    assert report["ambiguous_mal_rows"] == 2
    assert report["invalid_or_missing_mal_urls"] == 3
    assert len(catalog) == 1


def test_catalog_primary_keys_and_titles_are_critical():
    raw = pd.DataFrame({"animeID": [1, 1], "title": ["a", "b"], "mal_url": [None, None]})
    with pytest.raises(DataValidationError, match="duplicate primary"):
        prepare_catalog(raw)
    raw = raw.iloc[:1].copy()
    raw["title"] = " "
    with pytest.raises(DataValidationError, match="title"):
        prepare_catalog(raw)


def test_selection_has_complete_histories_in_unsorted_source():
    source = ratings([[user, anime, 8] for anime in range(1, 6) for user in range(1, 31)])
    chunks = [source.iloc[i:i+13] for i in range(0, len(source), 13)]
    selected, scanned = sample_complete_users(iter(chunks), 7, 42)
    expected_users = sorted(range(1, 31), key=lambda u: (hashlib.sha256(f"42:{u}".encode()).digest(), u))[:7]
    assert set(selected["userID"]) == set(expected_users)
    assert selected.groupby("userID").size().eq(5).all()
    assert scanned == len(source)
    shuffled = source.sample(frac=1, random_state=99)
    other, _ = sample_complete_users([shuffled], 7, 42)
    pd.testing.assert_frame_equal(selected, other)
    different, _ = sample_complete_users([source], 7, 43)
    assert set(different["userID"]) != set(selected["userID"])


def test_cached_rerun_is_identical_and_never_calls_network(tmp_path, monkeypatch):
    source = ratings([[1, 1, 8], [2, 2, 0], [1, 1, 8], [3, 1, 7]])
    source_bytes = source.to_csv(index=False).encode()
    catalog = pd.DataFrame({"animeID": [1, 2], "title": ["a", "b"],
        "mal_url": ["https://myanimelist.net/anime/431", "https://myanimelist.net/anime/1"]})
    catalog_bytes = catalog.to_csv(index=False).encode()
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/training_data.json").read_text())
    config.update(sample_users=2, chunk_rows=2)
    for label, body in [("catalog", catalog_bytes), ("ratings", source_bytes)]:
        config["source"][label + "_bytes"] = len(body)
        config["source"][label + "_sha256"] = hashlib.sha256(body).hexdigest()

    def get(url, **kwargs):
        response = MagicMock()
        response.raw = io.BytesIO(catalog_bytes if "animes.csv" in url else source_bytes)
        response.__enter__.return_value = response
        return response

    network = MagicMock(side_effect=get)
    monkeypatch.setattr("anime_recommender.ingestion.training_data.requests.get", network)
    first = run_pipeline(config, tmp_path, allow_full_scan=True)
    assert network.call_count == 2
    network.side_effect = AssertionError("Network must not be used on a cached rerun")
    second = run_pipeline(config, tmp_path)
    assert first == second
    assert network.call_count == 2
    assert (tmp_path / "data/processed/clean_interactions.parquet").exists()
    # Detect cache corruption rather than silently re-downloading.
    (tmp_path / "data/raw/training_sample/sample_interactions.parquet").write_bytes(b"corrupt")
    with pytest.raises(DataValidationError, match="checksum"):
        run_pipeline(config, tmp_path)
    manifest = tmp_path / "data/raw/training_sample/manifest.json"
    manifest.write_text('{"parameters": {}}')
    with pytest.raises(DataValidationError, match="Invalid cache manifest"):
        run_pipeline(config, tmp_path)


def test_no_automatic_full_scan_and_config_validation(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/training_data.json").read_text())
    with pytest.raises(DataValidationError, match="allow-full-scan"):
        run_pipeline(config, tmp_path)
    failure = json.loads((tmp_path / "data/processed/data_quality_report.json").read_text())
    assert failure["status"] == "failed"
    assert "allow-full-scan" in failure["error"]
    config["output_dir"] = "../outside"
    with pytest.raises(DataValidationError, match="inside data"):
        validate_config(config, tmp_path)


@pytest.mark.parametrize("section,key,value", [
    (None, "sample_users", True), (None, "chunk_rows", 0),
    (None, "seed", "42"), (None, "cache_dir", None),
    ("source", "revision", None), ("source", "ratings_sha256", None),
    ("cleaning", "rating_min", "zero"), ("cleaning", "conflict_policy", "last"),
])
def test_config_type_errors_are_actionable(tmp_path, section, key, value):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/training_data.json").read_text())
    target = config if section is None else config[section]
    target[key] = value
    with pytest.raises(DataValidationError):
        validate_config(config, tmp_path)
