"""Synthetic checks for bounded downloads and identity mapping; no network."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("requests")
pytest.importorskip("dotenv")


@pytest.fixture
def research():
    path = Path(__file__).resolve().parents[1] / "scripts/research_training_dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mal_identity_comes_from_url_not_internal_id(research):
    catalog = pd.DataFrame({"animeID": [1, 12], "mal_url": [
        "https://myanimelist.net/anime/431", "https://myanimelist.net/anime/1"]})
    _, safe = research.catalog_mapping(catalog)
    assert dict(zip(safe["animeID"], safe["mal_id"])) == {1: 431, 12: 1}


def test_all_ambiguous_or_invalid_mappings_excluded(research):
    catalog = pd.DataFrame({"animeID": [1, 2, 3, 4], "mal_url": [
        "https://myanimelist.net/anime/431", "https://myanimelist.net/anime/431",
        "https://example.invalid/anime/123", "https://myanimelist.net/anime/1"]})
    _, safe = research.catalog_mapping(catalog)
    assert safe["animeID"].tolist() == [4]


def test_window_discards_partial_boundary_rows(research):
    frame = research.parse_window(b"partial\n1,12,8\n2,20,9\n3,", 10)
    assert frame.to_dict("records") == [
        {"userID": 1, "animeID": 12, "rating": 8},
        {"userID": 2, "animeID": 20, "rating": 9},
    ]


def test_ignored_range_stops_before_reading_body(research, monkeypatch):
    response = MagicMock(status_code=200)
    manager = MagicMock()
    manager.__enter__.return_value = response
    monkeypatch.setattr(research.requests, "get", lambda *args, **kwargs: manager)
    with pytest.raises(RuntimeError):
        research.download_bounded("https://example.invalid/file", 100, start=0)
    response.raw.read.assert_not_called()
