"""Interactive exploration uses synthetic data and mocked HTTP only."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("implicit")

from anime_recommender.ingestion.yummyanime import YummyAnimeAuthorizationError
from anime_recommender.preprocessing.implicit_feedback import positive_feedback
from anime_recommender.preprocessing.profile_signals import normalize_profile_history, prepare_profile_signals
from anime_recommender.preprocessing.sparse_matrix import build_train_matrix
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.recommendation_exploration import (
    ModelWorkspace, load_workspace, optional_profile_view, profile_view, validation_view,
)
from anime_recommender.training.pipeline import fit_models


ROOT = Path(__file__).resolve().parents[1]


def row(anime_id, mal, status=2, rating=0, favorite=False):
    return {"anime_id": anime_id, "title": "Synthetic", "remote_ids": {"myanimelist_id": mal},
        "user": {"rating": rating, "list": {"list": {"id": status}, "is_fav": favorite}}}


@pytest.fixture
def workspace():
    raw = pd.DataFrame([(u, i, 8) for u in range(1, 5) for i in range(1, 7)],
        columns=["dataset_user_id", "dataset_anime_id", "rating"])
    train = positive_feedback(raw)
    mapping = pd.DataFrame({"dataset_anime_id": range(1, 7), "myanimelist_id": range(101, 107)})
    matrix, users, items = build_train_matrix(train, mapping)
    context = pd.DataFrame({"dataset_user_id": [10, 11], "dataset_anime_id": [1, 2]})
    observed = pd.DataFrame({"dataset_user_id": [10, 10, 10, 11], "dataset_anime_id": [1, 2, 7, 2],
        "rating": [8, 0, 9, 8], "rating_raw": [8, 0, 9, 8]})
    targets = pd.DataFrame({"dataset_user_id": [10, 11], "dataset_anime_id": [3, 4]})
    inputs = SimpleNamespace(train=train, matrix=matrix, train_users=users, items=items,
        train_fingerprint="synthetic", dataset_config={}, train_hashes={},
        validation_context=context, observed_context=observed, validation_target=targets,
        validation_users=pd.DataFrame({"dataset_user_id": [10, 11]}))
    config = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    config["als"].update(factors=3, iterations=3)
    bundle = fit_models(inputs, config)
    catalog = pd.DataFrame({"dataset_anime_id": range(1, 9), "myanimelist_id": range(101, 109),
                            "title": [f"Anime {i}" for i in range(1, 9)]})
    bundle.als.fit = Mock(side_effect=AssertionError("Inference must not refit"))
    return ModelWorkspace(bundle, inputs, catalog)


def test_binary_signals_status_veto_missing_ratings_and_mal_mapping(workspace):
    rows = [row(901, 101, rating=5), row(902, 102, rating=0),
        row(903, 103, status=1, rating=10, favorite=True),
        row(904, 104, status=3, rating=10, favorite=True),
        row(905, 105, status=0, rating=0), row(906, 106, rating=2),
        row(907, 107, rating=None), row(908, None, rating=8),
        row(909, 108, rating="bad")]
    original = copy.deepcopy(rows)
    frame = normalize_profile_history(rows, [row(905, 105, status=4, favorite=True)])
    signals = prepare_profile_signals(frame, workspace.catalog, workspace.bundle.items)
    assert rows == original
    assert signals.positive_ids == [1, 2, 5]  # Yummy IDs never used as dataset IDs.
    assert signals.seen_ids == list(range(1, 9))
    assert signals.coverage.fold_in_interactions == 3
    assert signals.coverage.cold_or_unmapped_positive_records == 2
    assert signals.frame.positive.tolist() == [True, True, False, False, True, False, True, True, False]
    assert signals.frame.user_rating.iloc[1] == 0


def test_favorite_only_duplicate_context_and_ambiguous_mapping(workspace):
    rows = [row(901, 101), row(901, 101)]
    frame = normalize_profile_history(rows, [row(902, 102, status=4)])
    signals = prepare_profile_signals(frame, workspace.catalog, workspace.bundle.items)
    assert signals.positive_ids == [1, 2] and signals.seen_ids == [1, 2]
    assert len(signals.frame) == 3  # Preserve rows; binary IDs are deduplicated.
    bad = pd.concat([workspace.catalog, workspace.catalog.iloc[:1]])
    with pytest.raises(DataValidationError, match="Ambiguous mapping"):
        prepare_profile_signals(frame, bad, workspace.bundle.items)


@pytest.mark.parametrize("empty", [True, False])
def test_profile_fallback_is_explicit_and_keeps_all_seen_filters(workspace, empty):
    client = Mock()
    client.get_user.return_value = {"id": 123}
    client.get_lists.return_value = [] if empty else [row(901, 101, status=1, rating=9), row(902, 107)]
    client.get_favorites.return_value = []
    client.request_log = []
    before = workspace.bundle.als.item_factors.copy()
    view = profile_view(workspace, client, "Synthetic", top_k=3)
    assert view["fallback"] and view["recommendations"].model_name.eq("popularity").all()
    if not empty:
        assert 1 not in view["recommendations"].dataset_anime_id.tolist()
        assert view["coverage"].fold_in_interactions == 0
    np.testing.assert_array_equal(before, workspace.bundle.als.item_factors)
    client.get_lists.assert_called_once_with(123)
    client.get_favorites.assert_called_once_with(123)


def test_profile_warm_context_infers_without_training_and_excludes_unrated_seen(workspace):
    client = Mock()
    client.get_user.return_value = {"id": 123}
    client.get_lists.return_value = [row(901, 101), row(902, 102, status=1), row(903, 103, status=0)]
    client.get_favorites.return_value = []
    client.request_log = []
    before = workspace.bundle.als.item_factors.copy()
    view = profile_view(workspace, client, "Synthetic", top_k=3)
    assert not view["fallback"] and view["coverage"].fold_in_interactions == 1
    assert not set(view["recommendations"].dataset_anime_id) & {1, 2, 3}
    assert view["recommendations"].title.notna().all()
    np.testing.assert_array_equal(before, workspace.bundle.als.item_factors)


def test_optional_profile_no_requests_when_blank_and_stop_after_authorization_error(workspace, monkeypatch, tmp_path):
    constructor = Mock()
    monkeypatch.setattr("anime_recommender.recommendation_exploration.YummyAnimeClient", constructor)
    assert optional_profile_view(workspace)["state"] == "skipped"
    constructor.assert_not_called()
    monkeypatch.setenv("YUMMYANIME_APPLICATION_TOKEN", "synthetic-public-token")
    client = constructor.return_value
    client.get_user.side_effect = YummyAnimeAuthorizationError("HTTP 403", 403)
    client.request_log = [{"HTTP": 403}]
    view = optional_profile_view(workspace, "Synthetic", env_path=tmp_path / ".env")
    assert view["state"] == "error" and "403" in view["error"]
    client.get_lists.assert_not_called()
    client.get_favorites.assert_not_called()


def test_validation_user_k_original_history_and_targets_are_posthoc(workspace):
    first = validation_view(workspace, 10, top_k=2)
    assert first["context"].rating_raw.tolist() == [8, 0, 9]
    assert first["context"].is_warm.tolist() == [True, True, False]
    assert first["context"].title.tolist() == ["Anime 1", "Anime 2", "Anime 7"]
    assert first["recommendations"].groupby("model_name").size().eq(2).all()
    assert first["recommendations"].is_hit.equals(first["recommendations"].dataset_anime_id.eq(3))
    original = first["recommendations"].drop(columns="is_hit")
    workspace.inputs.validation_target["dataset_anime_id"] = [5, 6]
    changed = validation_view(workspace, 10, top_k=2)
    pd.testing.assert_frame_equal(original, changed["recommendations"].drop(columns="is_hit"))
    second = validation_view(workspace, 11, top_k=3)
    assert second["user_id"] == 11 and second["recommendations"].groupby("model_name").size().eq(3).all()
    assert 1 not in first["recommendations"].dataset_anime_id.tolist()
    assert 2 not in second["recommendations"].dataset_anime_id.tolist()
    with pytest.raises(DataValidationError, match="validation"):
        validation_view(workspace, 999)


def test_model_parameter_change_requires_explicit_retrain(workspace, monkeypatch, tmp_path):
    directory = tmp_path / "configs"; directory.mkdir()
    config = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    (directory / "recommendation_models.json").write_text(json.dumps(config))
    monkeypatch.setattr("anime_recommender.recommendation_exploration.load_training_inputs", lambda _: workspace.inputs)
    monkeypatch.setattr("anime_recommender.recommendation_exploration.load_models", lambda *args: workspace.bundle)
    run = Mock(side_effect=AssertionError("Must not implicitly train"))
    monkeypatch.setattr("anime_recommender.recommendation_exploration.run_experiment", run)
    with pytest.raises(DataValidationError, match="retrain=True"):
        load_workspace(tmp_path, factors=4)
    run.assert_not_called()
    workspace.inputs.catalog_sha256 = "catalog-hash"
    monkeypatch.setattr("anime_recommender.recommendation_exploration.file_sha256", lambda _: "catalog-hash")
    monkeypatch.setattr(pd, "read_parquet", lambda _: workspace.catalog)
    run.side_effect = None
    run.return_value = SimpleNamespace(bundle=workspace.bundle)
    load_workspace(tmp_path, factors=4, retrain=True)
    passed = run.call_args.args[0]
    assert passed["als"]["factors"] == 4
    assert passed["model_dir"].endswith("-notebook")
    assert run.call_args.kwargs == {"train": True}
