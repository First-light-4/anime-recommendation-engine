"""Hand-calculated binary metrics and tiny offline model/report integration."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("implicit")

from anime_recommender.evaluation.cohorts import COHORTS, assign_cohorts, cohort_label
from anime_recommender.evaluation.coverage import catalog_coverage
from anime_recommender.evaluation.dashboard import dashboard_tables
from anime_recommender.evaluation.ranking_metrics import (
    hit_rate_at_k, map_at_k, ndcg_at_k, precision_at_k, ranking_metrics, recall_at_k,
)
from anime_recommender.evaluation.ranking_evaluator import evaluate_models, model_fingerprint, run_evaluation, validate_ks
from anime_recommender.evaluation.reports import load_report, save_report
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.implicit_feedback import positive_feedback
from anime_recommender.preprocessing.sparse_matrix import build_train_matrix
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.training.artifacts import save_models
from anime_recommender.training.dataset import run_pipeline
from anime_recommender.training.inputs import load_training_inputs
from anime_recommender.training.pipeline import fit_models


ROOT = Path(__file__).resolve().parents[1]


def test_independent_metric_examples():
    # Hits at ranks 1 and 3, with a third relevant item outside the list.
    ids, relevant = [11, 12, 13], {11, 13, 99}
    assert precision_at_k(ids, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ids, relevant, 3) == pytest.approx(2 / 3)
    assert ndcg_at_k(ids, relevant, 3) == pytest.approx(1.5 / (1 + 1 / math.log2(3) + .5))
    assert map_at_k(ids, relevant, 3) == pytest.approx((1 + 2 / 3) / 3)
    assert hit_rate_at_k(ids, relevant, 3) == 1
    assert hit_rate_at_k([12], relevant, 1) == 0
    assert map_at_k([11, 13], {11, 13, 99, 100}, 2) == 1  # min(K, |R|), not |R|.
    assert recall_at_k([11, 13], {11, 13, 99, 100}, 2) == .5


def test_empty_relevance_empty_lists_short_lists_and_cold_items():
    assert all(value is None for value in ranking_metrics([1], [], 5).values())
    assert all(value == 0 for value in ranking_metrics([], [1], 5).values())
    assert precision_at_k([1], [1], 10) == .1  # K remains 10, not list length.
    assert recall_at_k([1], [1], 10) == 1
    assert recall_at_k([1], [1, 999], 10) == .5
    assert ndcg_at_k([1], [1, 999], 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert map_at_k([1], [1, 999], 10) == .5
    assert catalog_coverage([1, 1, 2], [1, 2, 3, 4])["catalog_coverage"] == .5
    assert catalog_coverage([], [1])["catalog_coverage"] == 0
    assert catalog_coverage([], [])["catalog_coverage"] is None
    with pytest.raises(ValueError): catalog_coverage([99], [1])
    with pytest.raises(ValueError, match="Duplicate"): ranking_metrics([1, 2, 1], [1], 1)
    for k in (0, -1, True, 1.5):
        with pytest.raises(ValueError): ranking_metrics([], [], k)
    for ks in ([], [1, 1], [True], [0]):
        with pytest.raises(ValueError): validate_ks(ks)


@pytest.mark.parametrize("length,label", [(0, "0"), (1, "1"), (2, "2–4"), (4, "2–4"),
    (5, "5–9"), (9, "5–9"), (10, "10–19"), (19, "10–19"), (20, "20+"), (99, "20+")])
def test_cohort_boundaries(length, label):
    assert cohort_label(length) == label


@pytest.fixture
def tiny():
    train = positive_feedback(pd.DataFrame([(u, i, 8) for u in range(1, 5) for i in range(1, 7)],
        columns=["dataset_user_id", "dataset_anime_id", "rating"]))
    mapping = pd.DataFrame({"dataset_anime_id": range(1, 7), "myanimelist_id": range(101, 107)})
    matrix, train_users, items = build_train_matrix(train, mapping)
    def pairs(values):
        return pd.DataFrame(values, columns=["dataset_user_id", "dataset_anime_id"])
    inputs = SimpleNamespace(train=train, matrix=matrix, train_users=train_users, items=items,
        train_fingerprint="synthetic", train_hashes={}, dataset_config={"positive_threshold": 5},
        validation_users=pd.DataFrame({"dataset_user_id": [10, 11, 12, 13, 14]}),
        validation_context=pairs([(10, 1), (10, 999), (11, 998), (13, 3)]),
        observed_context=pairs([(10, 1), (10, 2), (10, 999), (11, 998), (13, 3)]),
        validation_target=pairs([(10, 3), (10, 997), (11, 4), (13, 777), (14, 5)]))
    config = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    config["als"].update(factors=3, iterations=3)
    bundle = fit_models(inputs, config)
    bundle.als.fit = Mock(side_effect=AssertionError("No retraining during evaluation"))
    return bundle, inputs


def test_full_warm_fallback_candidates_and_empty_groups(tiny):
    bundle, inputs = tiny
    report = evaluate_models(bundle, inputs, ks=[1, 10])
    metrics = report.tables["metrics"]
    def row(name, mode):
        return metrics.loc[metrics.model_name.eq(name) & metrics["mode"].eq(mode) & metrics.k.eq(10)].iloc[0]
    assert row("als", "full").recall_eligible_users == 4
    assert row("als", "warm").recall_eligible_users == 3  # No warm-context eligibility filter.
    assert row("als", "full").users_without_positive_targets == 1
    assert row("als", "full").users_without_warm_context == 3
    assert row("als", "full").cold_targets == 2
    assert row("als_fallback", "full").recall > row("als", "full").recall
    recs = report.tables["recommendations"]
    assert not recs.loc[recs.dataset_user_id.eq(10)].dataset_anime_id.isin([1, 2, 999]).any()
    assert set(recs.dataset_anime_id) <= set(bundle.items.dataset_anime_id)
    assert not recs.duplicated(["dataset_user_id", "model_name", "dataset_anime_id"]).any()
    statuses = report.tables["statuses"]
    fallback = statuses.loc[statuses.model_name.eq("als_fallback")].set_index("dataset_user_id")
    assert fallback.loc[11, "selected_model"] == "popularity" and fallback.loc[11, "fallback_used"]
    empty_group = report.tables["cohorts"].loc[lambda f: f.cohort.eq("20+")]
    assert empty_group.users.eq(0).all() and empty_group.recall.isna().all()
    assert empty_group.recall_eligible_users.eq(0).all()
    full = report.tables["user_metrics"].loc[lambda f: f["mode"].eq("full")]
    assert full.loc[full.dataset_user_id.eq(12), "precision"].isna().all()
    assert set(report.tables["history_distribution"].cohort) == set(COHORTS)


def test_canonical_warm_includes_zero_warm_context_unlike_legacy(tiny):
    from anime_recommender.evaluation.recommendation_checks import evaluate_validation
    bundle, inputs = tiny
    canonical = evaluate_models(bundle, inputs, ks=[10], include_als_fallback=False)
    legacy = evaluate_validation(bundle.recommenders(), inputs.validation_context,
        inputs.observed_context, inputs.validation_target, inputs.validation_users, inputs.items, 10)
    modern = canonical.tables["metrics"].query('model_name == "als" and mode == "warm"').iloc[0]
    old = legacy["summary"].query('model_name == "als"').iloc[0]
    # Users 11/14 have warm targets but no warm context: canonical counts their
    # unavailable ALS recommendations as zeros. Legacy excludes both from Warm.
    assert modern.recall_eligible_users == 3
    assert modern.recall == pytest.approx(1/3)
    assert old.warm_recall_users == 1
    assert old.warm_recall_at_k == 1
    for model in ("als", "popularity"):
        full = canonical.tables["metrics"].loc[lambda f: f.model_name.eq(model) & f["mode"].eq("full")].iloc[0]
        historic = legacy["summary"].loc[lambda f: f.model_name.eq(model)].iloc[0]
        assert full.recall == pytest.approx(historic.full_recall_at_k)
        assert full.ndcg == pytest.approx(historic.ndcg_at_k)


def test_canonical_warm_ignores_legacy_dataset_eligibility_flag(tiny):
    bundle, inputs = tiny
    before = evaluate_models(bundle, inputs, ks=[10])
    inputs.validation_users["warm_recall_eligible"] = False
    after = evaluate_models(bundle, inputs, ks=[10])
    pd.testing.assert_frame_equal(before.tables["metrics"], after.tables["metrics"])
    pd.testing.assert_frame_equal(before.tables["recommendations"], after.tables["recommendations"])


def test_targets_do_not_change_generation_or_cohorts_and_results_repeat(tiny):
    bundle, inputs = tiny
    before_items, before_users = bundle.als.item_factors.copy(), bundle.als.user_factors.copy()
    first = evaluate_models(bundle, inputs, ks=[1, 10])
    repeated = evaluate_models(bundle, inputs, ks=[1, 10])
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], repeated.tables[name])
    inputs.validation_target.loc[inputs.validation_target.dataset_user_id.eq(10), "dataset_anime_id"] = [4, 996]
    changed = evaluate_models(bundle, inputs, ks=[1, 10])
    pd.testing.assert_frame_equal(first.tables["recommendations"], changed.tables["recommendations"])
    pd.testing.assert_frame_equal(first.tables["users"][["dataset_user_id", "warm_context", "cohort"]],
                                  changed.tables["users"][["dataset_user_id", "warm_context", "cohort"]])
    assert first.metadata["dataset_fingerprint"] != changed.metadata["dataset_fingerprint"]
    np.testing.assert_array_equal(before_items, bundle.als.item_factors)
    np.testing.assert_array_equal(before_users, bundle.als.user_factors)
    inputs.validation_target.iloc[0, inputs.validation_target.columns.get_loc("dataset_anime_id")] = 1
    with pytest.raises(DataValidationError, match="separation"):
        evaluate_models(bundle, inputs)


def test_metric_k_prefix_and_report_roundtrip(tiny, tmp_path):
    bundle, inputs = tiny
    first = evaluate_models(bundle, inputs, ks=[1, 3], include_als_fallback=False)
    assert set(first.tables["statuses"].model_name) == {"als", "popularity"}
    second = evaluate_models(bundle, inputs, ks=[1], include_als_fallback=False)
    pd.testing.assert_frame_equal(first.tables["metrics"].loc[lambda f: f.k.eq(1)].reset_index(drop=True),
                                  second.tables["metrics"].reset_index(drop=True))
    save_report(first, tmp_path)
    loaded = load_report(tmp_path)
    assert first.metadata == loaded.metadata
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], loaded.tables[name])
    document = (tmp_path / "report.json").read_text()
    assert 'NaN' not in document and 'null' in document
    view = dashboard_tables(first, k=1)
    assert list(view["cohort_recall"].index) == list(COHORTS)
    assert view["cohort_recall"].loc["20+"].isna().all()
    with pytest.raises(ValueError): dashboard_tables(first, k=100)
    manifest = json.loads(document)
    manifest["metrics"][0]["recall"] = 123
    atomic_json(tmp_path / "report.json", manifest)
    with pytest.raises(ValueError, match="disagree"): load_report(tmp_path)
    save_report(first, tmp_path)
    (tmp_path / "metrics.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_report(tmp_path)


def test_all_users_without_warm_context_and_no_positive_targets(tiny, tmp_path):
    bundle, inputs = tiny
    inputs.validation_context = inputs.validation_context.iloc[:0]
    inputs.validation_target = inputs.validation_target.iloc[:0]
    report = evaluate_models(bundle, inputs, ks=[20])
    assert report.tables["metrics"].recall.isna().all()
    assert report.tables["metrics"].precision.isna().all()
    assert report.tables["metrics"].recall_eligible_users.eq(0).all()
    assert report.tables["metrics"].als_fold_in_share.eq(0).all()
    assert report.tables["example_users"].empty
    assert report.tables["recommendations"].loc[lambda f: f.model_name.eq("als")].empty
    save_report(report, tmp_path)
    loaded = load_report(tmp_path)
    pd.testing.assert_frame_equal(report.tables["metrics"], loaded.tables["metrics"])


def test_cohort_target_mean_median_include_zero_target_users(tiny):
    bundle, inputs = tiny
    report = evaluate_models(bundle, inputs, ks=[10])
    cohorts = report.tables["cohorts"].loc[lambda f: f.model_name.eq("als") & f["mode"].eq("full")].set_index("cohort")
    # Context=0 users have target sizes [1,0,1], not the eligible-only [1,1].
    assert cohorts.loc["0", "positive_targets_mean"] == pytest.approx(2 / 3)
    assert cohorts.loc["0", "positive_targets_median"] == 1
    assert cohorts.loc["0", "users"] == 3
    assert cohorts.loc["0", "recall_eligible_users"] == 2
    # Context=1 users have full sizes [2,1], warm sizes [1,0].
    assert cohorts.loc["1", "positive_targets_mean"] == 1.5
    assert cohorts.loc["1", "positive_targets_median"] == 1.5
    assert cohorts.loc["1", "warm_targets_mean"] == .5
    assert cohorts.loc["1", "warm_targets_median"] == .5
    assert pd.isna(cohorts.loc["20+", "positive_targets_mean"])
    assert pd.isna(cohorts.loc["20+", "positive_targets_median"])
    view = dashboard_tables(report)
    assert view["cohort_target_sizes"].loc["1", "positive_targets_mean"] == 1.5


def test_generation_arguments_match_and_targets_are_not_passed(tiny, monkeypatch):
    bundle, inputs = tiny
    models = bundle.recommenders()
    for recommender in models.values():
        recommender.recommend = Mock(wraps=recommender.recommend)
    monkeypatch.setattr(bundle, "recommenders", lambda: models)
    evaluate_models(bundle, inputs, ks=[1, 5, 10])
    pop_calls = models["popularity"].recommend.call_args_list
    als_calls = models["als"].recommend.call_args_list
    assert len(pop_calls) == len(als_calls) == len(inputs.validation_users)
    for pop, als in zip(pop_calls, als_calls):
        assert pop.args == als.args and pop.kwargs == als.kwargs
        assert set(als.kwargs) == {"top_k", "seen_items"}
        assert als.kwargs["top_k"] == 10
    assert 2 in pop_calls[0].kwargs["seen_items"]  # Observed low/zero item excluded.
    assert 2 not in pop_calls[0].args[0]  # Not used as positive fold-in context.


def test_macro_average_is_not_micro_and_cold_only_warm_metrics_undefined(tiny):
    bundle, inputs = tiny
    report = evaluate_models(bundle, inputs, ks=[10])
    user_metrics = report.tables["user_metrics"]
    full = user_metrics.loc[user_metrics.model_name.eq("popularity") & user_metrics["mode"].eq("full")]
    # Full lists cover all available candidates. Four target-bearing users:
    # Recall [1/2, 1, 0, 1], macro=.625; micro=3/5=.6.
    assert full.set_index("dataset_user_id").loc[10, "recall"] == .5
    summary = report.tables["metrics"].loc[lambda f: f.model_name.eq("popularity") & f["mode"].eq("full")].iloc[0]
    assert summary.recall == .625 and summary.recall != .6
    assert summary.precision == pytest.approx((.1 + .1 + 0 + .1) / 4)
    warm = user_metrics.loc[user_metrics.model_name.eq("popularity") & user_metrics["mode"].eq("warm")]
    assert pd.isna(warm.set_index("dataset_user_id").loc[13, "recall"])
    assert pd.isna(warm.set_index("dataset_user_id").loc[13, "ndcg"])
    assert full.set_index("dataset_user_id").loc[13, "recall"] == 0


def test_explicit_model_fingerprint_tracks_content_not_evaluation_targets(tiny):
    bundle, inputs = tiny
    first = evaluate_models(bundle, inputs, ks=[1])
    fingerprint = first.metadata["model_fingerprint"]
    assert len(fingerprint) == 64
    assert fingerprint == model_fingerprint(bundle)
    inputs.validation_target = inputs.validation_target.iloc[:0]
    second = evaluate_models(bundle, inputs, ks=[1])
    assert second.metadata["model_fingerprint"] == fingerprint
    assert second.metadata["dataset_fingerprint"] != first.metadata["dataset_fingerprint"]
    bundle.als.item_factors = bundle.als.item_factors.copy()
    bundle.als.item_factors[0, 0] += .01
    assert model_fingerprint(bundle) != fingerprint


def test_runner_does_not_use_test_files_or_refit(tmp_path, monkeypatch):
    processed = tmp_path / "data/processed"; processed.mkdir(parents=True)
    source = pd.DataFrame([(u, i, 8) for u in range(1, 21) for i in range(1, 7)],
        columns=["dataset_user_id", "dataset_anime_id", "rating"])
    source["myanimelist_id"] = source.dataset_anime_id + 100
    mapping = pd.DataFrame({"dataset_anime_id": range(1, 7), "myanimelist_id": range(101, 107)})
    catalog = mapping.assign(title=["Synthetic " + str(i) for i in range(1, 7)])
    tables = {"clean_interactions.parquet": source, "anime_catalog.parquet": catalog, "anime_id_mapping.parquet": mapping}
    for name, frame in tables.items(): atomic_parquet(frame, processed / name)
    atomic_json(processed / "data_quality_report.json", {"status": "success",
        "output_sha256": {name: file_sha256(processed / name) for name in tables}})
    run_pipeline(json.loads((ROOT / "configs/ml_dataset.json").read_text()), tmp_path)
    model_config = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    model_config["als"].update(factors=3, iterations=3)
    bundle = fit_models(load_training_inputs(processed / "ml"), model_config)
    save_models(bundle, tmp_path / "artifacts/models/are006")
    for path in (processed / "ml").glob("test_*"): path.write_bytes(b"test must never be opened")
    config_dir = tmp_path / "configs"; config_dir.mkdir()
    config = json.loads((ROOT / "configs/evaluation.json").read_text())
    atomic_json(config_dir / "evaluation.json", config)
    monkeypatch.setattr("anime_recommender.training.als.AlternatingLeastSquares.fit", Mock(side_effect=AssertionError("No fitting")))
    monkeypatch.setattr("requests.get", Mock(side_effect=AssertionError("No network")))
    report = run_evaluation(tmp_path)
    assert report.metadata["test_evaluated"] is False
    assert report.metadata["split"] == "validation"
    assert set(report.metadata["validation_files_sha256"]) == {
        "validation_context.parquet", "validation_target.parquet", "validation_observed_context.parquet", "validation_user_mapping.parquet"}
    assert (tmp_path / report.metadata["report_dir"] / "report.json").exists()
    from anime_recommender.evaluation.franchise_evaluation import run_franchise_evaluation
    baseline_manifest = tmp_path / report.metadata["report_dir"] / "report.json"
    original_hash = file_sha256(baseline_manifest)
    for name in ("franchise_evaluation.json", "franchise_filter.json", "franchise_relations.json", "yummyanime_franchise_relations.json"):
        if (ROOT / "configs" / name).exists():
            atomic_json(config_dir / name, json.loads((ROOT / "configs" / name).read_text()))
    monkeypatch.setattr("requests.post", Mock(side_effect=AssertionError("No network in post-processing")))
    result = run_franchise_evaluation(tmp_path)
    assert result["metadata"]["test_evaluated"] is False
    assert result["metadata"]["covered_train_items"] == 0
    assert result["metadata"]["removed_original_top_k"] == 0
    assert file_sha256(baseline_manifest) == original_hash
    config["split"] = "test"
    atomic_json(config_dir / "evaluation.json", config)
    with pytest.raises(ValueError, match="validation"):
        run_evaluation(tmp_path)


def test_franchise_diagnostic_preserves_predictions_when_targets_change(tiny):
    from anime_recommender.evaluation.franchise_evaluation import compare_franchise_filter
    from anime_recommender.preprocessing.franchise_mapping import build_franchise_mapping
    bundle, inputs = tiny
    snapshot = {"schema_version": 1, "nodes": [{"mal_id": i} for i in range(101, 107)],
        "relations": [{"source_mal_id": 101, "target_mal_id": 102, "relation": "SEQUEL"},
                      {"source_mal_id": 102, "target_mal_id": 103, "relation": "SEQUEL"}],
        "reviewed_policy": {"mainline_mal_ids": [101, 102, 103], "entry_mal_ids": [101]}}
    mapping = build_franchise_mapping(snapshot)
    catalog = bundle.items[["dataset_anime_id", "myanimelist_id"]]
    config = {"enabled": True, "max_recommendations_per_franchise": 1, "candidate_multiplier": 1}
    first = compare_franchise_filter(bundle, inputs, catalog, mapping, config, top_k=2)
    inputs.validation_target.loc[inputs.validation_target.dataset_user_id.eq(10), "dataset_anime_id"] = [5, 996]
    changed = compare_franchise_filter(bundle, inputs, catalog, mapping, config, top_k=2)
    pd.testing.assert_frame_equal(first["filtered_recommendations"], changed["filtered_recommendations"])
    pd.testing.assert_frame_equal(first["original_recommendations"], changed["original_recommendations"])
    assert first["metadata"]["model_fingerprint"] == changed["metadata"]["model_fingerprint"]
    assert first["metadata"]["dataset_fingerprint"] != changed["metadata"]["dataset_fingerprint"]
    assert "No completed proof" in first["metadata"]["completion_policy"]
