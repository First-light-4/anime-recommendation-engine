"""Structured synthetic graphs and bounded snapshots; no real API in tests."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from anime_recommender.ingestion.anime_relations import fetch_anilist_relations, relation_snapshot
from anime_recommender.inference.franchise_filter import filter_candidates, recommend_with_franchise
from anime_recommender.inference.recommendations import RecommendationResult
from anime_recommender.preprocessing.franchise_mapping import build_franchise_mapping, load_franchise_resources


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def graph():
    snapshot = {"schema_version": 1, "nodes": [{"mal_id": v} for v in range(1, 10)],
        "relations": [{"source_mal_id": a, "target_mal_id": b, "relation": kind} for a, b, kind in [
            (1, 2, "SEQUEL"), (2, 3, "SEQUEL"), (2, 6, "ALTERNATIVE"),
            (1, 4, "SIDE_STORY"), (1, 5, "SPIN_OFF"), (1, 8, "CHARACTER")]],
        "reviewed_policy": {"mainline_mal_ids": [1, 2, 3, 6, 9], "entry_mal_ids": [1, 9]}}
    return build_franchise_mapping(snapshot)


def candidates(ids):
    return pd.DataFrame({"dataset_anime_id": [1000 + i for i in ids], "myanimelist_id": ids,
        "score": [1 - n / 100 for n in range(len(ids))], "rank": range(1, len(ids) + 1),
        "model_name": ["als"] * len(ids)})


def test_exact_mal_and_dataset_seen_independent_of_status(graph):
    source = candidates([7, 8, 9])
    filtered = filter_candidates(source, graph, observed_mal_ids=[7], observed_dataset_ids=[1008])
    assert filtered.recommendations.myanimelist_id.tolist() == [9]
    assert filtered.diagnostics["excluded_observed"] == 2


def test_new_user_entry_point_and_franchise_cap(graph):
    source = candidates([3, 2, 1, 4, 5, 7, 8])
    filtered = filter_candidates(source, graph, top_k=5)
    assert filtered.recommendations.myanimelist_id.tolist() == [1, 7, 8]
    assert filtered.diagnostics["excluded_missing_prerequisite"] == 2
    assert filtered.diagnostics["excluded_franchise_duplicates"] == 2
    assert graph.franchise.get(8) is None  # Shared character does not imply franchise.


def test_first_and_late_completed_seasons_and_optional_parts(graph):
    after_first = filter_candidates(candidates([3, 2, 4, 7]), graph, [1], [1], max_per_franchise=2)
    assert after_first.recommendations.myanimelist_id.tolist() == [2, 4, 7]
    late = filter_candidates(candidates([2, 1, 4, 5, 7, 3]), graph, [3], [3], max_per_franchise=10)
    assert late.recommendations.myanimelist_id.tolist() == [4, 5, 7]
    assert late.diagnostics["excluded_earlier_seasons"] == 2
    # Planned/watching must not imply completion of missing earlier parts.
    observed_only = filter_candidates(candidates([3, 2, 1]), graph, [2], [])
    assert observed_only.recommendations.myanimelist_id.tolist() == [1]
    # Completing an optional movie/side story does not mark mainline watched.
    movie_only = filter_candidates(candidates([2, 1, 5]), graph, [4], [4])
    assert movie_only.recommendations.myanimelist_id.tolist() == [1]


def test_unknown_and_missing_ids_never_use_title_similarity(graph):
    source = candidates([7, 8])
    source["title"] = ["Season 9 of same title", "Season 10 of same title"]
    source.loc[1, "myanimelist_id"] = pd.NA
    result = filter_candidates(source, graph)
    assert result.recommendations.dataset_anime_id.tolist() == [1007, 1008]
    assert result.diagnostics["unknown_relationships"] == 2


def test_scores_order_no_mutation_and_no_network(graph, monkeypatch):
    monkeypatch.setattr("requests.get", Mock(side_effect=AssertionError("No API in filtering")))
    monkeypatch.setattr("requests.post", Mock(side_effect=AssertionError("No API in filtering")))
    source = candidates([3, 2, 1, 7])
    before = source.copy(deep=True)
    result = filter_candidates(source, graph)
    assert result.recommendations.myanimelist_id.tolist() == [1, 7]
    assert result.recommendations["rank"].tolist() == [1, 2]
    np.testing.assert_array_equal(result.recommendations.score, source.score.iloc[[2, 3]])
    pd.testing.assert_frame_equal(source, before)


def test_pool_expansion_and_exhaustion_never_reinsert_forbidden(graph):
    source = candidates([3, 2, 1, 7])
    model = Mock()
    model.model_name = "als"
    model.recommend.side_effect = lambda context, top_k, **kwargs: RecommendationResult(source.head(top_k), True, {})
    output = recommend_with_franchise(model, source, [], graph, top_k=2, candidate_multiplier=1)
    assert output.diagnostics["candidate_pool_attempts"] == [2, 4]
    assert output.recommendations.myanimelist_id.tolist() == [1, 7]
    assert all(call.kwargs["exclude_seen"] is False for call in model.recommend.call_args_list)
    exhausted = recommend_with_franchise(model, source, [], graph, top_k=5)
    assert len(exhausted.recommendations) == 2 and exhausted.diagnostics["insufficient_candidates"]
    assert exhausted.diagnostics["pool_exhausted"]


def test_empty_als_context_stays_unavailable_for_caller_fallback(graph):
    source = candidates([])
    model = Mock(); model.model_name = "als"
    model.recommend.return_value = RecommendationResult(source, False, {}, "no_warm_positive_context")
    result = recommend_with_franchise(model, pd.DataFrame({"myanimelist_id": [1]}), [], graph)
    assert result.recommendations.empty
    assert result.diagnostics["model_available"] is False
    assert result.diagnostics["model_unavailable_reason"] == "no_warm_positive_context"
    assert model.recommend.call_count == 1


def test_real_snapshot_aot_demon_alternatives_and_spinoffs():
    config, mapping = load_franchise_resources(ROOT)
    if 16498 not in mapping.franchise or 38000 not in mapping.franchise:
        pytest.skip("Optional local public metadata smoke test; synthetic unit tests need no snapshots")
    assert config["max_recommendations_per_franchise"] == 1
    aot_new = filter_candidates(candidates([25777, 16498]), mapping)
    assert aot_new.recommendations.myanimelist_id.tolist() == [16498]
    aot_late = filter_candidates(candidates([25777, 16498, 25781]), mapping, [38524], [38524])
    assert aot_late.recommendations.myanimelist_id.tolist() == [25781]  # No Regrets OVA not inferred seen.
    aot_next = filter_candidates(candidates([35760, 25777]), mapping, [16498], [16498])
    assert aot_next.recommendations.myanimelist_id.tolist() == [25777]
    demon = filter_candidates(candidates([51019, 47778, 49926, 38000]), mapping, [40456], [40456])
    assert demon.recommendations.myanimelist_id.tolist() == [47778]
    assert mapping.groups[40456] == mapping.groups[49926]  # Alternative prerequisite branches.
    assert mapping.viewing_path(47778) == ((38000,), (40456, 49926), (47778,))
    assert mapping.viewing_path(25777) == ((16498,), (25777,))
    assert mapping.viewing_path(25781) is None  # Optional OVA is not a forced prerequisite.
    # Other full franchises are intentionally unknown, not classified by title.
    assert mapping.franchise.get(5114) is None


def test_cycles_ambiguous_order_and_invalid_ids(graph):
    snapshot = json.loads(json.dumps(graph.snapshot))
    snapshot["relations"].append({"source_mal_id": 3, "target_mal_id": 2, "relation": "SEQUEL"})
    with pytest.raises(ValueError, match="Cyclic"): build_franchise_mapping(snapshot)
    snapshot = json.loads(json.dumps(graph.snapshot))
    snapshot["relations"].append({"source_mal_id": 9, "target_mal_id": 2, "relation": "SEQUEL"})
    ambiguous = build_franchise_mapping(snapshot)
    assert ambiguous.ancestors(2) is None
    assert len(filter_candidates(candidates([2]), ambiguous).recommendations) == 1
    with pytest.raises(ValueError): filter_candidates(candidates([1]), graph, [], [1])
    with pytest.raises(ValueError): filter_candidates(candidates([1, 1]), graph)


def test_ingestion_is_bounded_and_keeps_structured_provenance(monkeypatch):
    media = [{"idMal": 1, "format": "TV", "relations": {"edges": [
        {"relationType": "SEQUEL", "node": {"idMal": 2, "type": "ANIME", "format": "TV"}},
        {"relationType": "ADAPTATION", "node": {"idMal": 1, "type": "MANGA", "format": "MANGA"}}]}}]
    response = Mock(); response.json.return_value = {"data": {"Page": {"media": media}}}
    post = Mock(return_value=response); monkeypatch.setattr("requests.post", post)
    assert fetch_anilist_relations([1]) == media
    assert post.call_count == 1
    for ids in ([], list(range(1, 22)), [False]):
        with pytest.raises(ValueError): fetch_anilist_relations(ids)
    assert post.call_count == 1
    snapshot = relation_snapshot(media, [1])
    assert len(snapshot["relations"]) == 1 and len(snapshot["nodes"]) == 2
    assert snapshot["source_url"] == "https://graphql.anilist.co"
    assert snapshot["nodes"][1]["queried"] is False
