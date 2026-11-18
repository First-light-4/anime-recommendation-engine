"""Synthetic saved bundles and session cache; no live profile or network calls."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from anime_recommender.inference.model_versions import load_version_workspace
from anime_recommender.inference.profile_session import ProfileSession
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.implicit_feedback import positive_feedback
from anime_recommender.preprocessing.sparse_matrix import build_train_matrix
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.training.artifacts import load_models, save_models
from anime_recommender.training.pipeline import fit_models
from anime_recommender.training.scaling_experiments import add_baseline_deltas, fixed_validation_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_versions(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("No network")))
    (tmp_path / "configs").mkdir()
    registry = json.loads((ROOT / "configs/model_versions.json").read_text())
    config = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    config["als"].update(factors=3, iterations=3)
    atomic_json(tmp_path / "configs/model_versions.json", registry)
    atomic_json(tmp_path / "configs/recommendation_models.json", config)
    franchise = json.loads((ROOT / "configs/franchise_filter.json").read_text())
    franchise["enabled"] = False
    atomic_json(tmp_path / "configs/franchise_filter.json", franchise)
    catalog = pd.DataFrame({"dataset_anime_id": range(1, 9), "myanimelist_id": range(101, 109),
                            "title": [f"Synthetic {i}" for i in range(1, 9)]})
    path = tmp_path / config["catalog_path"]
    path.parent.mkdir(parents=True)
    atomic_parquet(catalog, path)
    atomic_json(path.parent / "data_quality_report.json", {"status": "success", "output_sha256": {path.name: file_sha256(path)}})
    for version, count in (("1k", 4), ("5k", 6), ("10k", 8)):
        train = positive_feedback(pd.DataFrame([(u, i, 8.) for u in range(1, 6) for i in range(1, count+1)],
            columns=["dataset_user_id", "dataset_anime_id", "rating"]))
        matrix, users, items = build_train_matrix(train, catalog[["dataset_anime_id", "myanimelist_id"]])
        inputs = SimpleNamespace(train=train, items=items, matrix=matrix, train_users=users,
            train_fingerprint="synthetic-"+version, train_hashes={}, dataset_config={"positive_threshold": 5})
        bundle = fit_models(inputs, config)
        bundle.metadata["model_version"] = version
        save_models(bundle, tmp_path / registry["versions"][version]["model_dir"])
    monkeypatch.setattr("implicit.cpu.als.AlternatingLeastSquares.fit", Mock(side_effect=AssertionError("No training on switching")))
    return tmp_path, registry


def source_row(mal, status=2):
    return {"anime_id": 9000+mal, "title": "Synthetic", "remote_ids": {"myanimelist_id": mal},
            "user": {"rating": 0, "list": {"list": {"id": status}, "is_fav": False}}}


def test_approved_default_loads_10k_and_never_falls_back(saved_versions):
    root, registry = saved_versions
    assert registry["default_version"] == "10k"
    assert load_version_workspace(root).bundle.metadata["model_version"] == "10k"
    client = Mock(request_log=[])
    client.get_user.return_value = {"id": 123}
    client.get_lists.return_value = [source_row(101)]
    client.get_favorites.return_value = []
    result = ProfileSession(root, client_factory=lambda: client).recommend("Synthetic")
    assert result["model_version"] == "10k"
    (root / registry["versions"]["10k"]["model_dir"] / "metadata.json").unlink()
    with pytest.raises(DataValidationError, match="10k.*missing"):
        load_version_workspace(root)
    assert load_version_workspace(root, "1k").bundle.als is not None
    assert load_version_workspace(root, "5k").bundle.als is not None


def test_switching_remaps_same_source_without_refetch_or_refit(saved_versions):
    root, _ = saved_versions
    client = Mock(request_log=[])
    client.get_user.return_value = {"id": 123}
    source = [source_row(101), source_row(106), source_row(102, 1), source_row(103, 0)]
    original = deepcopy(source)
    client.get_lists.return_value = source
    client.get_favorites.return_value = []
    factory = Mock(return_value=client)
    session = ProfileSession(root, client_factory=factory)
    first = session.recommend("Synthetic", "1k", 3)
    cached_source = session._sources["Synthetic"]["history"].copy(deep=True)
    assert first["coverage"].fold_in_interactions == 1
    for version, count in (("10k", 8), ("5k", 6), ("1k", 4)):
        view = session.recommend("Synthetic", version, 3)
        assert len(session.workspace(version).bundle.items) == count
        assert view["model_version"] == version
        assert not set(view["recommendations"].dataset_anime_id) & {1, 2, 3, 6}
        assert view["coverage"].fold_in_interactions == (1 if version == "1k" else 2)
        assert set(view["recommendations"].dataset_anime_id) <= set(range(1, count+1))
        assert_frame_equal(session._sources["Synthetic"]["history"], cached_source)
    short = session.recommend("Synthetic", "10k", 1)
    assert len(short["recommendations"]) == 1
    factory.assert_called_once()
    client.get_user.assert_called_once_with("Synthetic")
    client.get_lists.assert_called_once_with(123)
    client.get_favorites.assert_called_once_with(123)
    assert source == original


def test_missing_unknown_and_wrong_version_never_fall_back(saved_versions):
    root, registry = saved_versions
    with pytest.raises(DataValidationError, match="model_version"):
        load_version_workspace(root, "other")
    path = root / registry["versions"]["10k"]["model_dir"] / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["model_version"] = "5k"
    atomic_json(path, metadata)
    with pytest.raises(DataValidationError, match="version differs"):
        load_version_workspace(root, "10k")
    path.unlink()
    factory = Mock(side_effect=AssertionError("No profile API if model missing"))
    with pytest.raises(DataValidationError, match="missing"):
        ProfileSession(root, client_factory=factory).recommend("Synthetic", "10k")
    factory.assert_not_called()
    assert load_version_workspace(root, "1k").bundle.als is not None


def test_mapping_and_configuration_fingerprints(saved_versions):
    root, registry = saved_versions
    first = load_version_workspace(root, "1k")
    second = load_version_workspace(root, "5k")
    path = root / registry["versions"]["5k"]["model_dir"]
    with pytest.raises(DataValidationError, match="mapping"):
        load_models(path, expected_mapping=first.bundle.items)
    with pytest.raises(DataValidationError, match="different training"):
        load_models(path, expected_train_fingerprint="wrong")
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["training_config"]["top_k"] = 999
    atomic_json(path / "metadata.json", metadata)
    with pytest.raises(DataValidationError, match="configuration"):
        load_version_workspace(root, "5k")


def test_title_requests_only_initial_display(saved_versions, monkeypatch):
    root, _ = saved_versions
    client = Mock(request_log=[])
    client.get_user.return_value = {"id": 1}
    client.get_lists.return_value = [source_row(101)]
    client.get_favorites.return_value = []
    session = ProfileSession(root, client_factory=lambda: client)
    enrich = Mock(side_effect=lambda profile, env: {**profile, "title_cache": {104: "Название"}})
    monkeypatch.setattr("anime_recommender.inference.profile_session.enrich_profile_titles", enrich)
    for version in ("1k", "5k", "10k", "1k"):
        profile = session.recommend("Synthetic", version, 2)
        session.initial_titles(profile, "Synthetic")
    enrich.assert_called_once()
    assert session.title_cache[104] == "Название"


def test_comparison_deltas_and_fixed_identity_ignore_only_cold_flags():
    comparison = add_baseline_deltas(pd.DataFrame({"model_version": ["1k", "5k", "10k"],
        "als_full_recall_10": [.2, .3, .4], "als_full_ndcg_10": [.4, .5, .6]}))
    assert comparison.recall_delta_relative.iloc[1] == pytest.approx(.5)
    assert comparison.ndcg_delta_absolute.iloc[2] == pytest.approx(.2)
    frame = pd.DataFrame({"dataset_user_id": [1], "dataset_anime_id": [2], "rating": [8.], "is_cold_item": [True]})
    data = SimpleNamespace(observed_context=frame, validation_context=frame,
        validation_target=frame, validation_users=pd.DataFrame({"dataset_user_id": [1]}))
    before = fixed_validation_identity(data)
    frame.loc[:, "is_cold_item"] = False
    assert fixed_validation_identity(data) == before
    frame.loc[:, "rating"] = 7
    assert fixed_validation_identity(data) != before


def test_selective_scaling_reader_uses_shared_validation_and_never_reads_test(tmp_path, monkeypatch):
    from anime_recommender.training.inputs import load_training_inputs
    from anime_recommender.training.scaling import save_scaling_dataset, synthetic_scaling_demo
    data = synthetic_scaling_demo()["synthetic_20"]
    data.report["source_sha256"] = {"anime_catalog.parquet": "synthetic-catalog-sha"}
    shared = tmp_path / "shared"
    shared.mkdir()
    for cohort in ("validation", "test"):
        for part in ("context", "target"):
            key = cohort + "_observed_" + part
            atomic_parquet(data.tables[key], shared / (key + ".parquet"))
    directory = tmp_path / "expanded"
    save_scaling_dataset(directory, data, {"seed": 42}, baseline_directory=shared)
    (directory / "test_target.parquet").write_bytes(b"deliberately corrupt and must not be read")
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert not path.name.startswith("test_"), "Reader touched test"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    inputs = load_training_inputs(directory)
    assert inputs.matrix.shape == data.matrix.shape
    assert_frame_equal(inputs.observed_context, data.tables["validation_observed_context"])
    reference = tmp_path / "reference"
    reference.mkdir()
    atomic_json(reference / "dataset_report.json", {"status": "success", "dataset_version": 2,
        "baseline_reference": str(directory), "baseline_report_sha256": file_sha256(directory / "dataset_report.json")})
    assert load_training_inputs(reference).train_fingerprint == inputs.train_fingerprint
    # Missing/modified shared validation fails, rather than bypassing fingerprints.
    (shared / "validation_observed_context.parquet").write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="checksum"):
        load_training_inputs(directory)


def test_comparison_reuses_baseline_and_then_all_models_without_test_or_network(tmp_path, monkeypatch):
    from anime_recommender.training.inputs import load_training_inputs
    from anime_recommender.training.scaling import save_scaling_dataset, synthetic_scaling_demo
    from anime_recommender.training import scaling_experiments
    (tmp_path / "configs").mkdir()
    registry = json.loads((ROOT / "configs/model_versions.json").read_text())
    common = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    common["als"].update(factors=3, iterations=3)
    atomic_json(tmp_path / "configs/model_versions.json", registry)
    atomic_json(tmp_path / "configs/recommendation_models.json", common)
    catalog_path = tmp_path / common["catalog_path"]
    catalog_path.parent.mkdir(parents=True)
    atomic_parquet(pd.DataFrame({"dataset_anime_id": range(1, 9), "myanimelist_id": range(101, 109),
        "title": [f"Synthetic {i}" for i in range(1, 9)]}), catalog_path)
    for version, data in zip(("1k", "5k", "10k"), synthetic_scaling_demo().values()):
        data.report["source_sha256"] = {"anime_catalog.parquet": file_sha256(catalog_path)}
        save_scaling_dataset(tmp_path / registry["versions"][version]["dataset_dir"], data, {"version": version})
    entry = registry["versions"]["1k"]
    inputs = load_training_inputs(tmp_path / entry["dataset_dir"])
    save_models(fit_models(inputs, common), tmp_path / entry["model_dir"])
    fit = Mock(wraps=scaling_experiments.fit_models)
    monkeypatch.setattr(scaling_experiments, "fit_models", fit)
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("No HTTP")))
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert not path.name.startswith("test_"), "Comparison read test"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    first, first_examples = scaling_experiments.run_comparison(tmp_path, train_new=True)
    assert fit.call_count == 2
    assert first.reused_saved_model.tolist() == [True, False, False]
    files = list((tmp_path / "artifacts/models").rglob('*'))
    before = {str(p): file_sha256(p) for p in files if p.is_file()}
    second, second_examples = scaling_experiments.run_comparison(tmp_path, train_new=False)
    assert fit.call_count == 2
    assert second.reused_saved_model.all()
    assert_frame_equal(first_examples, second_examples)
    assert_frame_equal(first.drop(columns="reused_saved_model"), second.drop(columns="reused_saved_model"))
    assert before == {str(p): file_sha256(p) for p in files if p.is_file()}
