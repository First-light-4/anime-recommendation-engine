"""Source-index progression, unmapped barriers and optional parts, without HTTP."""

import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")

from anime_recommender.ingestion.anime_relations import fetch_yummy_viewing_orders
from anime_recommender.inference.franchise_filter import filter_candidates
from anime_recommender.preprocessing.franchise_mapping import build_franchise_mapping, load_franchise_resources
from anime_recommender.preprocessing.yummy_franchise_mapping import build_yummy_snapshot


ROOT = Path(__file__).resolve().parents[1]


def candidates(mals):
    return pd.DataFrame({"dataset_anime_id": [100000 + m for m in mals], "myanimelist_id": mals,
        "rank": range(1, len(mals) + 1), "score": [1 - i / 100 for i in range(len(mals))], "model_name": ["als"] * len(mals)})


@pytest.fixture
def source():
    roles = [(11, "адаптация манги", "tv"), (14, "дополнение", "movie"),
        (12, "продолжение", "tv"), (13, "продолжение", "tv"), (15, "ответвление сюжета", "tv")]
    rows = [{"anime_id": yummy, "title": "Same misleading title", "type": {"alias": kind},
        "data": {"id": 77, "index": idx, "text": text}} for idx, (yummy, text, kind) in enumerate(roles)]
    examples = [{"seed_yummy_id": 11, "viewing_order": list(reversed(rows))}]
    resolved = [{"anime_id": yummy, "myanimelist_id": yummy + 90} for yummy, _, _ in roles]
    return examples, resolved


@pytest.mark.parametrize("completed,expected", [([], 101), ([101], 102), ([101, 102], 103)])
def test_next_step_from_source_not_score_or_array_order(source, completed, expected):
    mapping = build_franchise_mapping(build_yummy_snapshot(*source))
    result = filter_candidates(candidates([103, 104, 102, 105, 101]), mapping,
        observed_mal_ids=completed, completed_mal_ids=completed)
    assert result.recommendations.myanimelist_id.tolist() == [expected]
    assert result.recommendations.score.iloc[0] == candidates([103, 104, 102, 105, 101]).set_index("myanimelist_id").loc[expected, "score"]


def test_late_completed_with_gaps_and_nonmandatory_extras(source):
    mapping = build_franchise_mapping(build_yummy_snapshot(*source))
    result = filter_candidates(candidates([102, 101, 104, 105]), mapping, [103], [103], max_per_franchise=2)
    assert result.recommendations.myanimelist_id.tolist() == [104, 105]
    assert result.diagnostics["excluded_earlier_seasons"] == 2
    assert mapping.snapshot["ordered_sequences"][0]["steps"] == [
        {"mal_ids": [101], "yummy_ids": [11], "source_index": 0},
        {"mal_ids": [102], "yummy_ids": [12], "source_index": 2},
        {"mal_ids": [103], "yummy_ids": [13], "source_index": 3}]


def test_planned_or_watching_not_completion(source):
    mapping = build_franchise_mapping(build_yummy_snapshot(*source))
    result = filter_candidates(candidates([103, 102, 101, 104]), mapping, [101], [])
    assert not set(result.recommendations.myanimelist_id) & {101, 102, 103}
    assert 104 in set(result.recommendations.myanimelist_id)  # Optional is not another season.


def test_next_missing_in_vocabulary_or_mal_mapping_never_bridges_gap(source):
    examples, resolved = source
    mapping = build_franchise_mapping(build_yummy_snapshot(examples, resolved))
    result = filter_candidates(candidates([103, 101]), mapping, [101], [101], train_mal_ids=[101, 103])
    assert result.recommendations.empty
    assert result.diagnostics["excluded_next_unavailable"] == 1
    missing = [row for row in resolved if row["anime_id"] != 12]
    mapping = build_franchise_mapping(build_yummy_snapshot(examples, missing))
    assert mapping.snapshot["ordered_sequences"][0]["steps"][1]["mal_ids"] == []
    assert filter_candidates(candidates([103]), mapping, [101], [101]).recommendations.empty
    # A trustworthy late completed part still withholds earlier items across that gap.
    assert filter_candidates(candidates([101]), mapping, [103], [103]).recommendations.empty


def test_unknown_role_conflicting_orders_and_duplicate_indices(source):
    examples, resolved = source
    changed = copy.deepcopy(examples)
    changed[0]["viewing_order"][2]["data"]["text"] = "unknown relationship"
    snapshot = build_yummy_snapshot(changed, resolved)
    assert snapshot["ordered_sequences"][0]["order_known"] is False
    unknown = filter_candidates(candidates([103]), build_franchise_mapping(snapshot))
    assert unknown.recommendations.myanimelist_id.tolist() == [103]
    assert unknown.diagnostics["unknown_relationships"] == 1
    duplicated = copy.deepcopy(examples)
    duplicated[0]["viewing_order"][1]["data"]["index"] = duplicated[0]["viewing_order"][0]["data"]["index"]
    with pytest.raises(ValueError): build_yummy_snapshot(duplicated, resolved)
    same = build_yummy_snapshot(examples + examples, resolved)
    assert len(same["ordered_sequences"]) == 1
    contradictory = copy.deepcopy(examples)
    contradictory[0]["viewing_order"][0]["data"]["text"] = "changed role"
    with pytest.raises(ValueError, match="Conflicting"):
        build_yummy_snapshot(examples + contradictory, resolved)


def test_real_aot_demon_and_cyberpunk_cases():
    _, mapping = load_franchise_resources(ROOT)
    if any(mal not in mapping.franchise for mal in (16498, 38000, 42310)):
        pytest.skip("Optional local source-data smoke test; no network or dataset required by unit tests")
    for completed, expected in [([], 16498), ([16498], 25777), ([16498, 25777], 35760), ([38524], 40028)]:
        result = filter_candidates(candidates([51535, 25777, 35760, 16498, 40028]), mapping, completed, completed)
        assert result.recommendations.myanimelist_id.tolist() == [expected]
    # A source compilation movie does NOT imply the entire first TV season.
    assert mapping.groups[23777] != mapping.groups[16498]
    for completed, expected in [([], 38000), ([38000], 49926), ([40456], 47778), ([49926], 47778)]:
        result = filter_candidates(candidates([51019, 49926, 47778, 40456, 38000]), mapping, completed, completed)
        assert result.recommendations.myanimelist_id.tolist() == [expected]
    assert filter_candidates(candidates([42310]), mapping).recommendations.myanimelist_id.tolist() == [42310]
    assert filter_candidates(candidates([42310]), mapping, [42310], [42310], train_mal_ids=[42310]).recommendations.empty
    assert mapping.sequence_state(42310, [42310])["next_mal_ids"] == [61990]


def test_offline_fetch_is_bounded_and_drops_user_fields(source):
    examples, _ = source
    client = Mock()
    client.get_anime.return_value = {"anime_id": 11, "viewing_order": examples[0]["viewing_order"], "user": {"private": "discard"}}
    client.get_anime_by_ids.return_value = [{"anime_id": x, "remote_ids": {"myanimelist_id": x + 90}, "user": {"private": "discard"}} for x in [11, 12, 13, 14, 15]]
    orders, resolved = fetch_yummy_viewing_orders(client, [11])
    assert len(orders) == 1 and len(resolved) == 5
    assert "user" not in repr(orders) and "private" not in repr(resolved)
    client.get_anime.assert_called_once_with(11)
    client.get_anime_by_ids.assert_called_once()
    with pytest.raises(ValueError): fetch_yummy_viewing_orders(client, [1, 2, 3, 4])


def test_yummy_detail_and_ids_methods_are_bounded_and_validated():
    from anime_recommender.ingestion.yummyanime import YummyAnimeClient
    client = YummyAnimeClient("synthetic")
    client._get = Mock(return_value={"anime_id": 11, "viewing_order": []})
    assert client.get_anime(11)["anime_id"] == 11
    assert client._get.call_args.args[0] == "/anime/11"
    for invalid in (0, -1, True, ""):
        with pytest.raises(ValueError): client.get_anime(invalid)
    client._get.return_value = []
    client.get_anime_by_ids([11, 12])
    assert "ids=11%2C12" in client._get.call_args.args[0]
    assert "anime-ID" in client._get.call_args.args[1]
    client.get_anime_collections(11, limit=1)
    assert client._get.call_args.args[0] == "/anime/11/collections?limit=1"
