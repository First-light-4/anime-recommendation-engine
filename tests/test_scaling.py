"""Offline regression tests for fixed evaluation and nested train scaling."""

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from anime_recommender.ingestion.training_data import sample_complete_users
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.training.scaling import (nested_users, prepare_scaling,
    save_scaling_dataset, load_scaling_dataset, synthetic_scaling_demo)


def test_nested_sampling_compatible_with_streaming_sampler():
    raw = pd.DataFrame([(u, 1, 8) for u in range(1, 41)], columns=["userID", "animeID", "rating"])
    baseline, _ = sample_complete_users([raw], 10, 42)
    maximum, _ = sample_complete_users([raw.iloc[:17], raw.iloc[17:]], 30, 42)
    middle = nested_users(maximum.userID, baseline.userID, 20, 42)
    assert set(baseline.userID) <= set(middle) <= set(maximum.userID)
    assert middle == nested_users(reversed(list(maximum.userID)), baseline.userID, 20, 42)
    direct, _ = sample_complete_users([raw], 20, 42)
    assert set(middle) == set(direct.userID)
    # Explicit baseline preservation also works for a legacy non-bottom-k population.
    assert 40 in nested_users(range(1, 41), [40], 5, 42)
    with pytest.raises(DataValidationError):
        nested_users([1], [1], 2, 42)


def test_fixed_holdouts_and_nested_train():
    data = synthetic_scaling_demo()
    baseline = data["synthetic_baseline"]
    for expanded in list(data.values())[1:]:
        assert set(baseline.tables["train_user_mapping"].dataset_user_id) <= set(expanded.tables["train_user_mapping"].dataset_user_id)
        assert expanded.report["checks"]["train_only_vocabulary"]
        for cohort in ("validation", "test"):
            assert_frame_equal(baseline.tables[cohort + "_observed_target"], expanded.tables[cohort + "_observed_target"])
            assert_frame_equal(baseline.tables[cohort + "_observed_context"], expanded.tables[cohort + "_observed_context"])
            ids = set(expanded.tables[cohort + "_user_mapping"].dataset_user_id)
            assert not ids & set(expanded.tables["train_user_mapping"].dataset_user_id)


def test_cold_becomes_warm_without_changing_targets():
    baseline = synthetic_scaling_demo()["synthetic_baseline"]
    config = baseline.report["config"]
    # Add an item seen only in fixed evaluation, then five NEW train users introduce it.
    from anime_recommender.training.dataset import prepare_dataset
    raw = pd.concat([baseline.tables["train_interactions"].drop(columns=["confidence", "feedback"], errors="ignore"),
                     *[baseline.tables[c + "_observed_" + p] for c in ("validation", "test") for p in ("context", "target")]])
    eval_id = baseline.tables["validation_user_mapping"].dataset_user_id.iloc[0]
    original_item = baseline.tables["validation_target"].dataset_anime_id.iloc[0]
    raw.loc[(raw.dataset_user_id.eq(eval_id)) & raw.dataset_anime_id.eq(original_item), "dataset_anime_id"] = 99
    mapping = pd.DataFrame({"dataset_anime_id": [*range(1, 9), 99], "myanimelist_id": [*range(101, 109), 999]})
    baseline = prepare_dataset(raw, mapping, config)
    new = pd.DataFrame([(u, i, 8.) for u in range(100, 105) for i in [1, 2, 3, 4, 99]], columns=raw.columns[:3])
    interactions = pd.concat([raw, new], ignore_index=True)
    expanded = prepare_scaling(interactions, mapping, baseline,
        [*baseline.tables["user_assignments"].dataset_user_id, *range(100, 105)], config)
    assert 99 not in set(baseline.tables["item_mapping"].dataset_anime_id)
    assert 99 in set(expanded.tables["item_mapping"].dataset_anime_id)
    assert expanded.context_matrices["validation"].shape[1] == len(expanded.tables["item_mapping"])
    for part in ("context", "target"):
        before, after = (d.tables["validation_" + part] for d in (baseline, expanded))
        assert_frame_equal(before.drop(columns="is_cold_item"), after.drop(columns="is_cold_item"))
        assert not after.loc[after.dataset_anime_id.eq(99), "is_cold_item"].any()


def test_save_load_idempotence_and_tampering(tmp_path):
    prepared = synthetic_scaling_demo()["synthetic_20"]
    directory = tmp_path / "experiment"
    manifest = {"seed": 42, "selected_user_ids": list(range(20))}
    loaded = save_scaling_dataset(directory, prepared, manifest)
    hashes = {p.name: p.read_bytes() for p in directory.iterdir()}
    again = save_scaling_dataset(directory, prepared, manifest)
    assert again.report == loaded.report
    assert hashes == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert (loaded.matrix != prepared.matrix).nnz == 0
    with pytest.raises(DataValidationError):
        save_scaling_dataset(directory, prepared, {"seed": 43})
    (directory / "train_matrix.npz").write_bytes(b"broken")
    with pytest.raises(DataValidationError, match="checksum"):
        load_scaling_dataset(directory)


def test_incompatible_version_and_threshold():
    baseline = synthetic_scaling_demo()["synthetic_baseline"]
    config = dict(baseline.report["config"], positive_threshold=7)
    with pytest.raises(DataValidationError, match="threshold"):
        prepare_scaling(pd.DataFrame(), pd.DataFrame(), baseline, range(1, 11), config)
    baseline.report["dataset_version"] = 1
    with pytest.raises(DataValidationError, match="version"):
        prepare_scaling(pd.DataFrame(), pd.DataFrame(), baseline, range(1, 11), dict(config, positive_threshold=5))


def test_reject_changed_fixed_history():
    baseline = synthetic_scaling_demo()["synthetic_baseline"]
    frames = [baseline.tables["train_interactions"].drop(columns=["positive", "confidence"])]
    frames += [baseline.tables[c + "_observed_" + p] for c in ("validation", "test") for p in ("context", "target")]
    raw = pd.concat(frames, ignore_index=True)
    uid = baseline.tables["validation_user_mapping"].dataset_user_id.iloc[0]
    raw.loc[raw.dataset_user_id.eq(uid), "rating"] = 9
    with pytest.raises(DataValidationError, match="changed"):
        prepare_scaling(raw, baseline.tables["item_mapping"], baseline,
            baseline.tables["user_assignments"].dataset_user_id, baseline.report["config"])


def test_shared_artifacts_and_baseline_reference(tmp_path):
    from scipy.sparse import save_npz
    from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
    from anime_recommender.training.scaling import register_baseline
    data = synthetic_scaling_demo()
    baseline = data["synthetic_baseline"]
    source = tmp_path / "original"
    source.mkdir()
    for name, frame in baseline.tables.items():
        atomic_parquet(frame, source / (name + ".parquet"))
    save_npz(source / "train_matrix.npz", baseline.matrix)
    for name, matrix in baseline.context_matrices.items():
        save_npz(source / (name + "_context_matrix.npz"), matrix)
    baseline.report["artifact_sha256"] = {p.name: file_sha256(p) for p in source.iterdir()}
    atomic_json(source / "dataset_report.json", baseline.report)
    reference = register_baseline(source, tmp_path / "baseline")
    assert (reference.matrix != baseline.matrix).nnz == 0
    assert len(list((tmp_path / "baseline").iterdir())) == 1
    expanded = save_scaling_dataset(tmp_path / "expanded", data["synthetic_20"], {"seed": 42}, baseline_directory=source)
    assert len(expanded.report["shared_artifacts"]) == 4
    assert not (tmp_path / "expanded" / "validation_observed_target.parquet").exists()
    (source / "validation_observed_target.parquet").write_bytes(b"tampered")
    with pytest.raises(DataValidationError):
        load_scaling_dataset(tmp_path / "expanded")


def test_zero_positive_evaluation_users_retained_and_targets_not_in_vocabulary():
    baseline = synthetic_scaling_demo()["synthetic_baseline"]
    from anime_recommender.training.dataset import prepare_dataset
    train_ids = baseline.tables["user_assignments"].query('split == "train"').dataset_user_id
    evaluation_ids = set(baseline.tables["user_assignments"].dataset_user_id) - set(train_ids)
    rows = pd.DataFrame([(u, i, 8. if u in set(train_ids) else 0.)
                         for u in range(1, 11) for i in range(1, 9)],
                        columns=["dataset_user_id", "dataset_anime_id", "rating"])
    mapping = pd.DataFrame({"dataset_anime_id": range(1, 9), "myanimelist_id": range(101, 109)})
    baseline = prepare_dataset(rows, mapping, baseline.report["config"])
    result = prepare_scaling(rows, mapping, baseline, range(1, 11), baseline.report["config"])
    for cohort in ("validation", "test"):
        assert result.report["evaluation"][cohort]["users_without_positive_targets"] == 1
        assert result.report["evaluation"][cohort]["users_without_known_context"] == 1
        assert result.context_matrices[cohort].shape == (1, 8)
        assert result.context_matrices[cohort].nnz == 0


def test_acquisition_gate_and_reuse_without_network(tmp_path, monkeypatch):
    from anime_recommender.ingestion.training_data import acquire
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unexpected HTTP request"))
    config = {"source": {}, "cache_dir": "data/raw/sample", "sample_users": 30, "seed": 42}
    with pytest.raises(DataValidationError, match="approval"):
        acquire(config, tmp_path, allow_full_scan=False)


def test_local_pipeline_end_to_end_and_repeat(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from anime_recommender.ingestion import pipeline, scaling
    from anime_recommender.training import dataset
    root = Path(__file__).resolve().parents[1]
    source_config = json.loads((root / "configs/training_data.json").read_text())
    source_config["sample_users"] = 10
    raw = pd.DataFrame([(u, i, "8") for u in range(1, 31) for i in range(1, 9)],
                       columns=["userID", "animeID", "rating"])
    catalog = pd.DataFrame({"animeID": range(1, 9), "title": [f"Synthetic {i}" for i in range(1, 9)],
        "mal_url": [f"https://myanimelist.net/anime/{100+i}" for i in range(1, 9)]})
    provenance = {"parameters": {"source": source_config["source"]}, "cache_sha256": {},
        "source_ratings_sha256": source_config["source"]["ratings_sha256"],
        "source_ratings_bytes": source_config["source"]["ratings_bytes"]}
    def cached(config, root, allow_full_scan=False):
        assert not allow_full_scan
        return raw.loc[raw.userID.le(config["sample_users"])].copy(), catalog, provenance
    monkeypatch.setattr(pipeline, "acquire", cached)
    monkeypatch.setattr(scaling, "acquire", cached)
    pipeline.run_pipeline(source_config, tmp_path)
    ml_config = json.loads((root / "configs/ml_dataset.json").read_text())
    dataset.run_pipeline(ml_config, tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/source.json").write_text(json.dumps(source_config))
    config = {"source_config": "configs/source.json", "baseline_dir": "data/processed/ml",
        "output_dir": "data/processed/scaling", "cache_dir": "data/raw/scaling", "sizes": [20, 30]}
    (tmp_path / "configs/scaling.json").write_text(json.dumps(config))
    first = scaling.run_scaling(tmp_path, execute=True)
    second = scaling.run_scaling(tmp_path, execute=True)
    assert first == second
    assert [row["selected_users"] for row in first["statistics"]] == [10, 20, 30]
    from anime_recommender.ingestion.training_data import atomic_json
    source_config["seed"] = 123
    atomic_json(tmp_path / "configs/source.json", source_config)
    with pytest.raises(DataValidationError, match="configuration"):
        scaling.run_scaling(tmp_path, execute=True)


def test_aggregate_audit_detects_cold_flag_and_file_corruption(tmp_path):
    from anime_recommender.evaluation.scaling_validation import audit_scaling_datasets
    from anime_recommender.ingestion.training_data import file_sha256
    data = synthetic_scaling_demo()
    protected = tmp_path / "protected.txt"
    protected.write_text("unchanged")
    hashes = {"protected.txt": file_sha256(protected)}
    audit = audit_scaling_datasets(data, protected_sha256=hashes, root=tmp_path)
    assert audit["checks"]["protected_artifacts_unchanged"]
    assert len(audit["coverage"]) == 12
    assert len(audit["matrices"]) == 9
    assert audit["comparison"].selected_users.tolist() == [10, 20, 30]
    protected.write_text("changed")
    with pytest.raises(DataValidationError, match="Protected"):
        audit_scaling_datasets(data, protected_sha256=hashes, root=tmp_path)
    data["synthetic_30"].tables["validation_context"].loc[:, "is_cold_item"] = True
    with pytest.raises(DataValidationError, match="cold"):
        audit_scaling_datasets(data)


def test_aggregate_audit_serialization_has_no_user_histories(tmp_path):
    import json
    from anime_recommender.evaluation.scaling_validation import (
        audit_scaling_datasets, load_scaling_resources, save_scaling_audit)
    data = synthetic_scaling_demo()
    for name, dataset in data.items():
        dataset.report["scaling_fingerprint"] = name
    assert load_scaling_resources(tmp_path) == {"status": "not_recorded"}
    (tmp_path / "controlled_run_resources.json").write_text(json.dumps({
        "elapsed_seconds": 1, "peak_working_set_bytes": 123, "private_field": "omit"}))
    saved = save_scaling_audit(tmp_path, data, audit_scaling_datasets(data))
    assert saved["resources"] == {"elapsed_seconds": 1, "peak_working_set_bytes": 123}
    payload = (tmp_path / "aggregate_audit.json").read_text()
    assert "dataset_user_id" not in payload
    assert "selected_user_ids" not in payload
    assert "private_field" not in payload
