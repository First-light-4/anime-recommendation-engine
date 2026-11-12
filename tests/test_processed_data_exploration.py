"""Local-only artifact loading and execution of notebook 02 with synthetic tables."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
requests = pytest.importorskip("requests")

from anime_recommender.ingestion.processed_data import load_processed_data
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError


@pytest.fixture
def local_data(tmp_path):
    interactions = pd.DataFrame({"dataset_user_id": [1, 2], "dataset_anime_id": [12, 12],
                                "myanimelist_id": [431, 431], "rating": [0., 8.]})
    catalog = pd.DataFrame({"dataset_anime_id": [12], "myanimelist_id": [431], "title": ["Synthetic"]})
    mapping = catalog[["dataset_anime_id", "myanimelist_id"]]
    tables = {"clean_interactions.parquet": interactions, "anime_catalog.parquet": catalog,
              "anime_id_mapping.parquet": mapping}
    for name, frame in tables.items():
        atomic_parquet(frame, tmp_path / name)
    atomic_json(tmp_path / "data_quality_report.json", {"status": "success",
        "output_sha256": {name: file_sha256(tmp_path / name) for name in tables}})
    return tmp_path


def test_loader_never_downloads_and_checks_corruption(local_data, monkeypatch):
    get = Mock(side_effect=AssertionError("Unexpected network"))
    monkeypatch.setattr(requests, "get", get)
    data = load_processed_data(local_data)
    assert len(data.interactions) == 2
    get.assert_not_called()
    (local_data / "anime_id_mapping.parquet").write_bytes(b"corrupt")
    with pytest.raises(DataValidationError, match="checksum"):
        load_processed_data(local_data)


def test_loader_rejects_failed_report(local_data):
    atomic_json(local_data / "data_quality_report.json", {"status": "failed"})
    with pytest.raises(DataValidationError, match="successful"):
        load_processed_data(local_data)


def test_training_notebook_executes_locally_with_mock_profile(local_data, monkeypatch):
    pytest.importorskip("dotenv")
    pytest.importorskip("IPython.display")
    original_loader = load_processed_data
    monkeypatch.setattr("anime_recommender.ingestion.processed_data.load_processed_data",
                        lambda path: original_loader(local_data))
    monkeypatch.setattr("IPython.display.display", lambda *args: None)
    monkeypatch.setenv("YUMMYANIME_APPLICATION_TOKEN", "synthetic-token")
    monkeypatch.setenv("YUMMYANIME_TEST_NICKNAME", "synthetic-profile")
    def response(payload):
        result = Mock(status_code=200)
        result.json.return_value = {"response": payload}
        return result
    get = Mock(side_effect=[response({"id": 1}), response([
        {"anime_id": 999, "remote_ids": {"myanimelist_id": 431}}
    ])])
    monkeypatch.setattr(requests, "get", get)
    path = Path(__file__).resolve().parents[1] / "notebooks/02_training_data_exploration.ipynb"
    namespace = {}
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    assert get.call_count == 2
    assert namespace["stats"]["summary"]["records"] == 2
    assert namespace["coverage"]["catalog_coverage"] == 1
    assert namespace["coverage"]["training_coverage"] == 1
