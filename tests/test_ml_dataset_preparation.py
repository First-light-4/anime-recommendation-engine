"""Synthetic ML preparation tests: no model fitting and no network."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
requests = pytest.importorskip("requests")

from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.implicit_feedback import positive_feedback
from anime_recommender.preprocessing.data_split import assign_users, filter_train, split_context_target, split_observed_interactions
from anime_recommender.preprocessing.sparse_matrix import build_train_matrix
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.evaluation.dataset_validation import validate_dataset
from anime_recommender.training.dataset import prepare_dataset, run_pipeline, load_dataset, validate_config


def config():
    path = Path(__file__).resolve().parents[1] / "configs/ml_dataset.json"
    return json.loads(path.read_text())


def rows(pairs):
    return pd.DataFrame(pairs, columns=["dataset_user_id", "dataset_anime_id", "rating"])


def mapping(ids):
    return pd.DataFrame({"dataset_anime_id": ids, "myanimelist_id": [1000 + v for v in ids]})


@pytest.mark.parametrize("threshold,expected", [(5, [5, 6, 7, 8, 10]), (6, [6, 7, 8, 10]), (7, [7, 8, 10]), (8, [8, 10])])
def test_feedback_thresholds_missing_zero_and_source_unchanged(threshold, expected):
    source = rows([(1, i + 1, value) for i, value in enumerate([0, None, 5, 6, 7, 8, 10])])
    original = source.copy(deep=True)
    positives = positive_feedback(source, threshold)
    assert positives["rating"].tolist() == expected
    assert positives["positive"].eq(1).all() and positives["confidence"].eq(1).all()
    pd.testing.assert_frame_equal(source, original)


def test_feedback_rejects_bad_values_duplicates_and_arbitrary_weights():
    with pytest.raises(DataValidationError, match="invalid ratings"):
        positive_feedback(rows([(1, 1, "bad")]))
    with pytest.raises(DataValidationError, match="duplicate"):
        positive_feedback(rows([(1, 1, 8), (1, 1, 8)]))
    with pytest.raises(DataValidationError, match="confidence"):
        positive_feedback(rows([(1, 1, 8)]), confidence=10)


def test_user_split_is_exhaustive_disjoint_and_order_independent():
    first = assign_users(range(1, 101), 42)
    other = assign_users(reversed(range(1, 101)), 42)
    pd.testing.assert_frame_equal(first, other)
    assert first["split"].value_counts().to_dict() == {"train": 80, "validation": 10, "test": 10}
    assert not first["dataset_user_id"].duplicated().any()
    assert not first.equals(assign_users(range(1, 101), 43))


def test_context_target_is_complete_disjoint_and_reproducible():
    positives = positive_feedback(rows([(1, i, 8) for i in range(1, 11)]))
    context, target = split_context_target(positives, [1, 2, 3], 42)
    shuffled = positives.sample(frac=1, random_state=10)
    other_context, other_target = split_context_target(shuffled, [1, 2, 3], 42)
    pd.testing.assert_frame_equal(context, other_context)
    pd.testing.assert_frame_equal(target, other_target)
    assert len(context) == 8 and len(target) == 2
    assert not set(context["dataset_anime_id"]) & set(target["dataset_anime_id"])
    assert set(context["dataset_anime_id"]) | set(target["dataset_anime_id"]) == set(range(1, 11))
    assert context["is_cold_item"].sum() + target["is_cold_item"].sum() == 7


def test_train_k_core_reaches_both_thresholds_after_cascade():
    positives = positive_feedback(rows([(1, 1, 8), (1, 2, 8), (2, 1, 8), (2, 2, 8),
                                       (3, 2, 8), (3, 3, 8)]))
    filtered, rounds = filter_train(positives, 2, 2)
    assert len(filtered) == 4 and set(filtered["dataset_user_id"]) == {1, 2}
    assert len(rounds) == 3
    assert sum(r["user_filter_removed_rows"] + r["item_filter_removed_rows"] for r in rounds) == 2


def cold_fixture():
    cfg = config()
    cfg.update(min_user_positives=2, min_item_users=2)
    assignments = assign_users(range(1, 11), cfg["seed"])
    train_users = set(assignments.loc[assignments["split"].eq("train"), "dataset_user_id"])
    source = rows([(u, i, 8) for u in range(1, 11) for i in ([1, 2] if u in train_users else [99, 100])])
    return cfg, source, mapping([1, 2, 99, 100, 101])


def test_vocabulary_is_train_only_even_for_popular_eval_items():
    cfg, source, ids = cold_fixture()
    prepared = prepare_dataset(source, ids, cfg)
    assert prepared.tables["item_mapping"]["dataset_anime_id"].tolist() == [1, 2]
    mutated = source.copy()
    mutated.loc[mutated["dataset_anime_id"].eq(99), "dataset_anime_id"] = 101
    second = prepare_dataset(mutated, ids, cfg)
    pd.testing.assert_frame_equal(prepared.tables["item_mapping"], second.tables["item_mapping"])
    pd.testing.assert_frame_equal(prepared.tables["train_interactions"], second.tables["train_interactions"])
    assert (prepared.matrix != second.matrix).nnz == 0
    for name in ("validation", "test"):
        report = prepared.report["evaluation"][name]
        assert report["positive_rows"] == 2 and report["cold_interaction_share"] == 1
        assert report["users_without_known_context"] == 1
        assert prepared.tables[name + "_target"]["is_cold_item"].all()
        assert prepared.context_matrices[name].shape == (1, 2)
        assert prepared.context_matrices[name].nnz == 0


def test_raw_holdout_independent_of_ratings_threshold_and_input_order():
    source = rows([(1, i, 0 if i % 2 else 8) for i in range(1, 11)])
    original = source.copy(deep=True)
    context, target = split_observed_interactions(source, 42)
    mutated = source.sample(frac=1, random_state=1).reset_index(drop=True)
    mutated["rating"] = 10
    other_context, other_target = split_observed_interactions(mutated, 42)
    assert context["dataset_anime_id"].tolist() == other_context["dataset_anime_id"].tolist()
    assert target["dataset_anime_id"].tolist() == other_target["dataset_anime_id"].tolist()
    pd.testing.assert_frame_equal(source, original)
    tiny_context, tiny_target = split_observed_interactions(source.iloc[:1], 42)
    assert len(tiny_context) == 1 and tiny_target.empty


def test_target_labels_cannot_change_train_context_or_user_retention():
    cfg, source, ids = cold_fixture()
    first = prepare_dataset(source, ids, cfg)
    mutated = source.copy()
    hidden_pairs = set()
    for name in ("validation", "test"):
        raw_target = first.tables[name + "_observed_target"]
        hidden_pairs.update(zip(raw_target.dataset_user_id, raw_target.dataset_anime_id))
    mask = [(u, i) in hidden_pairs for u, i in zip(mutated.dataset_user_id, mutated.dataset_anime_id)]
    mutated.loc[mask, "rating"] = 0
    second = prepare_dataset(mutated, ids, cfg)
    pd.testing.assert_frame_equal(first.tables["train_interactions"], second.tables["train_interactions"])
    pd.testing.assert_frame_equal(first.tables["item_mapping"], second.tables["item_mapping"])
    for name in ("validation", "test"):
        pd.testing.assert_frame_equal(first.tables[name + "_context"], second.tables[name + "_context"])
        assert (first.context_matrices[name] != second.context_matrices[name]).nnz == 0
        assert len(first.tables[name + "_user_mapping"]) == len(second.tables[name + "_user_mapping"]) == 1
        assert second.tables[name + "_target"].empty
        assert second.report["evaluation"][name]["users_without_positive_targets"] == 1


def test_zero_rating_eval_users_are_not_removed_by_activity_filter():
    cfg, source, ids = cold_fixture()
    assignments = assign_users(source.dataset_user_id, cfg["seed"])
    eval_ids = set(assignments.loc[assignments.split.ne("train"), "dataset_user_id"])
    source.loc[source.dataset_user_id.isin(eval_ids), "rating"] = 0
    result = prepare_dataset(source, ids, cfg)
    for name in ("validation", "test"):
        assert result.tables[name + "_context"].empty and result.tables[name + "_target"].empty
        assert len(result.tables[name + "_user_mapping"]) == 1
        assert result.report["evaluation"][name]["users"] == 1
        assert result.report["evaluation"][name]["users_without_known_context"] == 1
        assert result.context_matrices[name].nnz == 0


def test_sparse_matrix_coordinates_stable_mappings_and_no_duplicate_summing():
    train = positive_feedback(rows([(10, 12, 8), (3, 1, 8), (3, 12, 8)]))
    matrix, users, items = build_train_matrix(train, mapping([12, 1]))
    assert matrix.format == "csr" and matrix.shape == (2, 2) and matrix.nnz == 3
    assert str(matrix.dtype) == "float32"
    assert users["dataset_user_id"].tolist() == [3, 10]
    assert items["dataset_anime_id"].tolist() == [1, 12]
    assert items["myanimelist_id"].tolist() == [1001, 1012]
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1 and matrix[1, 1] == 1 and matrix[1, 0] == 0
    with pytest.raises(DataValidationError, match="Duplicate"):
        build_train_matrix(pd.concat([train, train]), mapping([12, 1]))


def test_validation_detects_injected_context_target_leakage():
    cfg, source, ids = cold_fixture()
    p = prepare_dataset(source, ids, cfg)
    evaluations = {name: (p.tables[name + "_context"], p.tables[name + "_target"]) for name in ("validation", "test")}
    expected = {name: pd.concat(pair) for name, pair in evaluations.items()}
    ctx, target = evaluations["validation"]
    evaluations["validation"] = (ctx, pd.concat([target, ctx]))
    with pytest.raises(DataValidationError, match="Context-target leakage"):
        validate_dataset(p.tables["train_interactions"], evaluations, p.tables["user_assignments"],
            p.tables["train_user_mapping"], p.tables["item_mapping"], p.matrix, expected, 2, 2)


def test_empty_and_too_small_data_fail_without_changing_thresholds():
    cfg = config()
    empty = rows([])
    assert positive_feedback(empty).empty
    with pytest.raises(DataValidationError, match="Not enough users"):
        prepare_dataset(empty, mapping([1]), cfg)
    with pytest.raises(DataValidationError, match="Train is empty"):
        prepare_dataset(rows([(u, 1, 8) for u in range(1, 11)]), mapping([1]), cfg)
    assert cfg["min_user_positives"] == 5 and cfg["min_item_users"] == 5


def test_empty_eval_cohorts_are_explicit_not_silently_rebalanced():
    cfg = config()
    assignments = assign_users(range(1, 11), cfg["seed"])
    training_users = set(assignments.loc[assignments["split"].eq("train"), "dataset_user_id"])
    source = rows([(u, i, 8 if u in training_users else 0) for u in range(1, 11) for i in range(1, 7)])
    result = prepare_dataset(source, mapping(list(range(1, 7))), cfg)
    assert len(result.report["warnings"]) == 4
    assert result.tables["validation_target"].empty
    assert result.tables["test_context"].empty
    assert str(result.tables["validation_user_mapping"]["dataset_user_id"].dtype) == "int64"
    assert result.report["cohorts"]["validation"]["allocated_users"] == 1
    assert len(result.tables["validation_user_mapping"]) == 1
    assert result.report["evaluation"]["validation"]["users_without_positive_targets"] == 1
    assert result.context_matrices["validation"].nnz == 0


def test_notebook_03_calls_modules_offline_without_persisting_source(tmp_path, monkeypatch):
    pytest.importorskip("IPython.display")
    from anime_recommender.ingestion.processed_data import ProcessedData
    cfg = config()
    source = rows([(u, i, 8) for u in range(1, 11) for i in range(1, 7)])
    source["myanimelist_id"] = source["dataset_anime_id"] + 1000
    ids = mapping(list(range(1, 7)))
    catalog = ids.assign(title="Synthetic")
    monkeypatch.setattr("anime_recommender.ingestion.processed_data.load_processed_data",
                        lambda path: ProcessedData(source, catalog, ids, {}))
    monkeypatch.setattr("anime_recommender.training.dataset.run_pipeline",
                        lambda config, root: prepare_dataset(source, ids, config))
    monkeypatch.setattr("IPython.display.display", lambda *args: None)
    get = Mock(side_effect=AssertionError("Notebook must be offline"))
    monkeypatch.setattr(requests, "get", get)
    path = Path(__file__).resolve().parents[1] / "notebooks/03_ml_dataset_preparation.ipynb"
    namespace = {}
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    assert namespace["prepared"].matrix.nnz == 48
    assert namespace["prepared"].report["checks"]["user_disjoint"] is True
    get.assert_not_called()


def test_pipeline_repeat_is_identical_and_local_artifacts_restore(tmp_path, monkeypatch):
    cfg = config()
    source = rows([(u, i, 8) for u in range(1, 11) for i in range(1, 7)])
    source["myanimelist_id"] = source["dataset_anime_id"] + 1000
    catalog = mapping(list(range(1, 7)))
    catalog["title"] = "Synthetic"
    directory = tmp_path / cfg["input_dir"]
    directory.mkdir(parents=True)
    tables = {"clean_interactions.parquet": source, "anime_catalog.parquet": catalog,
              "anime_id_mapping.parquet": catalog[["dataset_anime_id", "myanimelist_id"]]}
    for name, frame in tables.items():
        atomic_parquet(frame, directory / name)
    atomic_json(directory / "data_quality_report.json", {"status": "success",
        "output_sha256": {name: file_sha256(directory / name) for name in tables}})
    get = Mock(side_effect=AssertionError("No network allowed"))
    monkeypatch.setattr(requests, "get", get)
    first = run_pipeline(cfg, tmp_path)
    second = run_pipeline(cfg, tmp_path)
    assert first.report == second.report
    restored = load_dataset(tmp_path / cfg["output_dir"])
    assert (first.matrix != restored.matrix).nnz == 0
    for name in ("validation", "test"):
        assert (first.context_matrices[name] != restored.context_matrices[name]).nnz == 0
    for key, frame in first.tables.items():
        pd.testing.assert_frame_equal(frame, restored.tables[key])
    get.assert_not_called()
    with pytest.raises(DataValidationError, match="configuration differs"):
        load_dataset(tmp_path / cfg["output_dir"], expected_config=dict(cfg, positive_threshold=8))
    with pytest.raises(DataValidationError, match="source hashes differ"):
        load_dataset(tmp_path / cfg["output_dir"], expected_source_sha256={"changed": "source"})
    (tmp_path / cfg["output_dir"] / "train_matrix.npz").write_bytes(b"corrupt")
    with pytest.raises(DataValidationError, match="checksum"):
        load_dataset(tmp_path / cfg["output_dir"])


@pytest.mark.parametrize("key,value", [("positive_threshold", 4), ("confidence", 10),
    ("context_fraction", 1), ("train_fraction", .9), ("min_user_positives", 0),
    ("output_dir", "../outside"), ("seed", "42")])
def test_invalid_config_does_not_relax_policy(tmp_path, key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(DataValidationError):
        validate_config(cfg, tmp_path)
