"""Tiny CPU models, no real API, no external data and no test-split evaluation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("implicit")
pytest.importorskip("threadpoolctl")
pytest.importorskip("pyarrow")

from anime_recommender.preprocessing.implicit_feedback import positive_feedback
from anime_recommender.preprocessing.sparse_matrix import build_train_matrix
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.inference.als import ALSRecommender
from anime_recommender.inference.fold_in import fold_in
from anime_recommender.inference.popularity import PopularityRecommender
from anime_recommender.training.als import train_als
from anime_recommender.training.popularity import train_popularity
from anime_recommender.training.pipeline import fit_models, run_experiment
from anime_recommender.training.artifacts import save_models, load_models
from anime_recommender.evaluation.recommendation_checks import evaluate_validation, diagnostic_metrics
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.training.inputs import load_training_inputs


ROOT = Path(__file__).resolve().parents[1]


def config():
    result = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    result["als"].update(factors=3, iterations=6)
    return result


@pytest.fixture
def toy():
    source = pd.DataFrame([(1, 1, 8), (1, 2, 8), (2, 1, 8), (2, 2, 8),
        (2, 3, 8), (3, 2, 8), (3, 4, 8), (4, 3, 8), (4, 5, 8)],
        columns=["dataset_user_id", "dataset_anime_id", "rating"])
    train = positive_feedback(source)
    ids = pd.DataFrame({"dataset_anime_id": [1, 2, 3, 4, 5], "myanimelist_id": [101, 102, 103, 104, 105]})
    matrix, users, items = build_train_matrix(train, ids)
    return SimpleNamespace(train=train, matrix=matrix, train_users=users, items=items,
                           train_fingerprint="synthetic", dataset_config={"positive_threshold": 5}, train_hashes={})


def test_popularity_unique_train_users_ties_and_new_user(toy):
    counts = train_popularity(pd.concat([toy.train, toy.train.iloc[:1]]), toy.items)
    assert counts.unique_users.tolist() == [2, 3, 2, 1, 1]
    recommender = PopularityRecommender(counts, toy.items)
    assert recommender.recommend([], top_k=5).recommendations.dataset_anime_id.tolist() == [2, 1, 3, 4, 5]
    assert recommender.recommend([2, 999], top_k=10).recommendations.dataset_anime_id.tolist() == [1, 3, 4, 5]
    assert recommender.recommend([2], top_k=5, exclude_seen=False).recommendations.dataset_anime_id.tolist()[0] == 2


def test_als_orientation_finite_and_deterministic(toy):
    first = train_als(toy.matrix, config()["als"])
    second = train_als(toy.matrix, config()["als"])
    assert first.user_factors.shape == (4, 3) and first.item_factors.shape == (5, 3)
    assert np.isfinite(first.user_factors).all() and np.isfinite(first.item_factors).all()
    np.testing.assert_allclose(first.item_factors, second.item_factors, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(first.user_factors, second.user_factors, rtol=1e-5, atol=1e-6)
    with pytest.raises(DataValidationError, match="CSR"):
        train_als(toy.matrix.tocsc(), config()["als"])


@pytest.mark.parametrize("context,warm,cold", [([], 0, 0), ([999], 0, 1),
    ([1], 1, 0), ([1, 2], 2, 0), ([1, 1, 999], 1, 1)])
def test_fold_in_context_cases_and_no_retraining(toy, context, warm, cold):
    model = train_als(toy.matrix, config()["als"])
    old_items, old_users = model.item_factors.copy(), model.user_factors.copy()
    model.fit = Mock(side_effect=AssertionError("Do not refit"))
    model.partial_fit_users = Mock(side_effect=AssertionError("Do not update train users"))
    model.partial_fit_items = Mock(side_effect=AssertionError("Do not update items"))
    result = fold_in(model, context, toy.items)
    assert result.available == bool(warm)
    assert result.context_vector.shape == (1, 5) and result.context_vector.nnz == warm
    assert result.diagnostics["warm_context"] == warm and result.diagnostics["cold_context"] == cold
    np.testing.assert_array_equal(model.item_factors, old_items)
    np.testing.assert_array_equal(model.user_factors, old_users)
    model.fit.assert_not_called()
    model.partial_fit_users.assert_not_called()
    model.partial_fit_items.assert_not_called()


def test_fold_in_matches_regularized_least_squares_not_average(toy):
    model = train_als(toy.matrix, config()["als"])
    result = fold_in(model, [1, 3, 999], toy.items)
    y = model.item_factors.astype("float64")
    # alpha=1 and observed confidence=1: A=Y'Y+lambda I, b=sum observed y_i.
    expected = np.linalg.solve(y.T @ y + model.regularization * np.eye(3), y[[0, 2]].sum(axis=0))
    np.testing.assert_allclose(result.factor, expected, rtol=2e-5, atol=2e-6)
    assert not np.allclose(result.factor, y[[0, 2]].mean(axis=0))


def test_same_candidates_seen_low_rating_items_and_unknown_ids(toy):
    bundle = fit_models(toy, config())
    for recommender in bundle.recommenders().values():
        result = recommender.recommend([1, 999], top_k=10, seen_items=[1, 4, 9999])
        assert result.available
        assert set(result.recommendations.dataset_anime_id) == {2, 3, 5}
        assert result.recommendations["rank"].tolist() == [1, 2, 3]
        assert result.recommendations.score.is_monotonic_decreasing
    unavailable = ALSRecommender(bundle.als, toy.items).recommend([999], top_k=10)
    assert not unavailable.available and unavailable.reason == "no_warm_positive_context"
    assert unavailable.recommendations.empty
    with pytest.raises(ValueError):
        bundle.recommenders()["als"].recommend([True])
    with pytest.raises(ValueError):
        bundle.recommenders()["popularity"].recommend([-1])


def test_safe_save_load_mapping_guard_and_same_recommendations(toy, tmp_path):
    bundle = fit_models(toy, config())
    before = bundle.recommenders()
    save_models(bundle, tmp_path)
    with np.load(tmp_path / "als.npz", allow_pickle=False) as data:
        assert all(value.dtype.kind != "O" for value in data.values())
    loaded = load_models(tmp_path, toy.items, "synthetic")
    for name, recommender in loaded.recommenders().items():
        a = before[name].recommend([1, 999], top_k=3).recommendations
        b = recommender.recommend([1, 999], top_k=3).recommendations
        assert a.dataset_anime_id.tolist() == b.dataset_anime_id.tolist()
        np.testing.assert_allclose(a.score, b.score, rtol=1e-5, atol=1e-6)
    wrong = toy.items.copy()
    wrong["myanimelist_id"] += 100
    with pytest.raises(DataValidationError, match="different model"):
        load_models(tmp_path, wrong)
    with pytest.raises(DataValidationError, match="different training"):
        load_models(tmp_path, expected_train_fingerprint="other")
    (tmp_path / "popularity.parquet").write_bytes(b"corrupt")
    with pytest.raises(DataValidationError, match="checksum"):
        load_models(tmp_path)


def test_target_changes_metrics_not_factors_or_recommendations(toy):
    bundle = fit_models(toy, config())
    users = pd.DataFrame({"dataset_user_id": [100, 101, 102]})
    context = pd.DataFrame({"dataset_user_id": [100, 101], "dataset_anime_id": [1, 999]})
    target = pd.DataFrame({"dataset_user_id": [100, 100, 101], "dataset_anime_id": [3, 998, 2]})
    before = bundle.als.item_factors.copy()
    first = evaluate_validation(bundle.recommenders(), context, context, target, users, toy.items, 2)
    changed = target.copy();changed["dataset_anime_id"] = [4, 997, 3]
    second = evaluate_validation(bundle.recommenders(), context, context, changed, users, toy.items, 2)
    pd.testing.assert_frame_equal(first["recommendations"], second["recommendations"])
    np.testing.assert_array_equal(before, bundle.als.item_factors)
    als = first["summary"].set_index("model_name").loc["als"]
    assert als.users == 3 and als.full_recall_users == 2 and als.warm_recall_users == 1
    assert als.users_without_positive_targets == 1 and als.unavailable_users == 2
    no_target = first["per_user"].loc[lambda f: f.dataset_user_id.eq(102)]
    assert no_target.full_recall.isna().all()


def test_full_cold_denominator_ndcg_and_warm_conditional_metrics(toy):
    recs = pd.DataFrame({"dataset_user_id": [100], "model_name": ["als"], "dataset_anime_id": [1], "rank": [1]})
    statuses = pd.DataFrame({"dataset_user_id": [100, 101, 102], "model_name": ["als"] * 3,
                            "warm_context": [1, 0, 0], "als_available": [True, False, False]})
    target = pd.DataFrame({"dataset_user_id": [100, 100, 101], "dataset_anime_id": [1, 999, 3]})
    users = pd.DataFrame({"dataset_user_id": [100, 101, 102]})
    summary, per_user = diagnostic_metrics(recs, statuses, target, users, toy.items, 2)
    assert summary.iloc[0].full_recall_at_k == .25  # (.5 + 0)/2, not .5 from dropping cold/no-context
    assert summary.iloc[0].warm_recall_at_k == 1
    assert summary.iloc[0].ndcg_at_k == pytest.approx((1 / (1 + 1 / np.log2(3))) / 2)
    assert summary.iloc[0].full_recall_users == 2 and summary.iloc[0].warm_recall_users == 1


@pytest.fixture
def local_ml(tmp_path):
    from anime_recommender.training.dataset import run_pipeline
    source = pd.DataFrame([(u, i, 8) for u in range(1, 11) for i in range(1, 7)],
        columns=["dataset_user_id", "dataset_anime_id", "rating"])
    source["myanimelist_id"] = source.dataset_anime_id + 100
    ids = pd.DataFrame({"dataset_anime_id": range(1, 7), "myanimelist_id": range(101, 107)})
    catalog = ids.assign(title=["Synthetic " + str(i) for i in range(1, 7)])
    directory = tmp_path / "data/processed";directory.mkdir(parents=True)
    tables = {"clean_interactions.parquet": source, "anime_catalog.parquet": catalog, "anime_id_mapping.parquet": ids}
    for name, frame in tables.items():atomic_parquet(frame, directory / name)
    atomic_json(directory / "data_quality_report.json", {"status": "success",
        "output_sha256": {name: file_sha256(directory / name) for name in tables}})
    run_pipeline(json.loads((ROOT / "configs/ml_dataset.json").read_text()), tmp_path)
    return tmp_path


def test_selective_reader_never_opens_test_and_pipeline_loads_without_training(local_ml, monkeypatch):
    # If the loader reads or hashes any test artifact, it will fail on corruption.
    for path in (local_ml / "data/processed/ml").glob("test_*"):
        path.write_bytes(b"test must never be read in ARE-006")
    inputs = load_training_inputs(local_ml / "data/processed/ml")
    assert not set(inputs.train_users.dataset_user_id) & set(inputs.validation_users.dataset_user_id)
    result = run_experiment(config(), local_ml, train=True)
    assert len(result.diagnostics["summary"]) == 2
    monkeypatch.setattr("anime_recommender.training.pipeline.fit_models", Mock(side_effect=AssertionError("No retraining")))
    restored = run_experiment(config(), local_ml, train=False)
    pd.testing.assert_frame_equal(result.diagnostics["recommendations"], restored.diagnostics["recommendations"])
