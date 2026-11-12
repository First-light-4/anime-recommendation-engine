"""Synthetic API and list checks; never contact the real API."""

import copy
from unittest.mock import Mock

import pytest

requests = pytest.importorskip("requests")
pd = pytest.importorskip("pandas")

from anime_recommender.ingestion.yummyanime import (
    YummyAnimeClient, YummyAnimeError, YummyAnimeAuthorizationError, YummyAnimeNetworkError,
)
from anime_recommender.exploration import explore_profile
from anime_recommender.preprocessing.user_data import normalize_user_lists, status_distribution
from anime_recommender.evaluation.user_data import (
    compare_favorites, compare_rating_statistics, rating_summary, user_quality_report, website_comparison,
)
from anime_recommender.evaluation.training_data import map_user_to_training, training_statistics
from anime_recommender.preprocessing.training_data import DataValidationError


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = {"response": payload}
    return result


def row(anime=1, status=2, rating=8, favorite=True, mal=431):
    return {"anime_id": anime, "title": "Synthetic", "rating": 1.1,
        "user": {"rating": rating, "list": {"is_fav": favorite,
                 "list": None if status is None else {"id": status, "href": "watched"}}},
        "remote_ids": {"myanimelist_id": mal}}


def test_client_calls_only_public_header_and_expected_endpoints(monkeypatch):
    get = Mock(side_effect=[response({"id": 1}), response([]), response([]), response([])])
    monkeypatch.setattr(requests, "get", get)
    client = YummyAnimeClient("synthetic-token")
    assert client.get_user("a/b") == {"id": 1}
    assert client.get_lists(1) == []
    assert client.get_favorites(1) == []
    assert client.get_rating_statistics(1) == []
    assert get.call_args_list[0].args[0].endswith("/users/a%2Fb")
    assert [call.args[0].split("yani.tv")[1] for call in get.call_args_list[1:]] == [
        "/users/1/lists", "/users/1/lists/4", "/users/1/stats/ratings"]
    for call in get.call_args_list:
        assert call.kwargs["headers"] == {"X-Application": "synthetic-token", "Accept": "application/json"}
        assert call.kwargs["allow_redirects"] is False
        assert call.kwargs["timeout"] == (10, 30)
    assert "synthetic-token" not in repr(client)
    assert "a/b" not in str(client.request_log)


@pytest.mark.parametrize("status,error_type", [(401, YummyAnimeAuthorizationError),
    (403, YummyAnimeAuthorizationError), (404, YummyAnimeError), (429, YummyAnimeError),
    (302, YummyAnimeError), (500, YummyAnimeError)])
def test_http_error_stops_orchestration_without_disclosure(monkeypatch, status, error_type):
    get = Mock(return_value=response({"secret": "body"}, status))
    monkeypatch.setattr(requests, "get", get)
    client = YummyAnimeClient("private-looking-test-value")
    with pytest.raises(error_type) as error:
        client.get_user("synthetic-profile")
    assert error.value.status == status
    assert "synthetic-profile" not in str(error.value)
    assert "private-looking-test-value" not in str(error.value)
    get.reset_mock()
    result = explore_profile(client, "synthetic-profile")
    assert not result.complete and result.frame.empty
    assert get.call_count == 1


@pytest.mark.parametrize("error", [requests.Timeout("sensitive"), requests.ConnectionError("sensitive")])
def test_network_errors_are_sanitized(monkeypatch, error):
    get = Mock(side_effect=error)
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(YummyAnimeNetworkError, match="details omitted"):
        YummyAnimeClient("test-token").get_user("profile")
    assert get.call_count == 1


@pytest.mark.parametrize("payload", [None, {}, "bad", [1]])
def test_invalid_list_envelope(monkeypatch, payload):
    monkeypatch.setattr(requests, "get", Mock(return_value=response(payload)))
    with pytest.raises(YummyAnimeError):
        YummyAnimeClient("test-token").get_lists(1)


def test_invalid_json_and_api_error(monkeypatch):
    bad = response(None)
    bad.json.side_effect = ValueError("sensitive")
    monkeypatch.setattr(requests, "get", Mock(return_value=bad))
    with pytest.raises(YummyAnimeError, match="invalid JSON"):
        YummyAnimeClient("test-token").get_user("profile")
    bad.json.side_effect = None
    bad.json.return_value = {"error": "sensitive"}
    with pytest.raises(YummyAnimeError, match="envelope"):
        YummyAnimeClient("test-token").get_user("profile")


def test_normalization_keeps_semantics_and_raw_input():
    rows = [row(), row(anime=2, status=None, rating=0), row()]
    original = copy.deepcopy(rows)
    frame = normalize_user_lists(rows)
    assert rows == original
    assert len(frame) == 3
    assert frame["user_rating"].tolist() == [8, 0, 8]  # not global rating=1.1
    assert pd.isna(frame.loc[1, "list_id"])
    assert frame.loc[0, "myanimelist_id"] == 431
    assert status_distribution(frame).loc["No status", "API"] == 1
    quality = user_quality_report(rows, frame)["summary"]
    assert quality["duplicate_anime_ids"] == 1
    assert quality["missing_status_and_favorite"] == 1
    assert rating_summary(frame)["zero_ratings"] == 1


def test_missing_and_malformed_fields_are_explicit():
    rows = [row(anime="bad", status=1.5, mal=True), {}, row(anime=3, status=99, favorite="yes")]
    frame = normalize_user_lists(rows)
    quality = user_quality_report(rows, frame)
    assert frame["anime_id"].isna().sum() == 2
    assert frame["list_id"].isna().sum() == 2
    assert quality["coercions"]["list_id"] == 1
    assert quality["coercions"]["myanimelist_id"] == 1
    assert status_distribution(frame).loc["Other (99)", "API"] == 1
    assert quality["summary"]["invalid_favorite_values"] == 2


def test_empty_lists_and_statistics_have_stable_schema():
    frame = normalize_user_lists([])
    assert len(frame.columns) == 8
    assert rating_summary(frame)["rated_count"] == 0
    assert compare_favorites(frame, [])["favorite_membership_matches"] is True
    assert compare_rating_statistics(frame, [])["matches"] is True
    assert website_comparison(frame)["Match"].isna().all()
    assert len(user_quality_report([], frame)["columns"]) == 8


def test_checks_detect_favorite_and_rating_disagreements():
    frame = normalize_user_lists([row(), row(anime=2, rating=0, favorite=False)])
    assert compare_favorites(frame, [{"anime_id": 1}])["favorite_membership_matches"] is True
    assert compare_favorites(frame, [{"anime_id": 2}])["favorite_membership_matches"] is False
    assert compare_favorites(frame, [{"anime_id": None}])["favorite_membership_matches"] is False
    assert compare_rating_statistics(frame, [{"rating": 8, "count": 1}])["matches"] is True
    assert compare_rating_statistics(frame, [{"rating": 9, "count": 1}])["matches"] is False
    assert compare_rating_statistics(frame, [{"rating": 8, "count": -1}])["matches"] is None
    assert compare_rating_statistics(frame, [{"rating": 8, "count": .5}])["matches"] is None
    assert compare_favorites(normalize_user_lists([row(anime=None)]), [])["favorite_membership_matches"] is False


def test_mapping_joins_mal_not_source_id_and_distinguishes_active_items():
    frame = normalize_user_lists([row(anime=999, mal=431), row(anime=888, mal=1), row(anime=777, mal=9999)])
    mapping = pd.DataFrame({"dataset_anime_id": [1, 12], "myanimelist_id": [431, 1]})
    interactions = pd.DataFrame({"dataset_anime_id": [12]})
    joined, summary = map_user_to_training(frame, mapping, interactions)
    assert joined.loc[0, "dataset_anime_id"] == 1
    assert joined["known_in_training"].tolist() == [False, True, False]
    assert summary["catalog_matches"] == 2 and summary["training_matches"] == 1
    assert summary["catalog_coverage"] == pytest.approx(2/3)
    with pytest.raises(DataValidationError, match="Ambiguous"):
        map_user_to_training(frame, pd.concat([mapping, mapping]), interactions)


def test_training_sparsity_uses_unique_pairs():
    frame = pd.DataFrame({"dataset_user_id": [1, 1, 2], "dataset_anime_id": [1, 1, 2], "rating": [0, 0, 8]})
    stats = training_statistics(frame)
    assert stats["summary"]["sparsity"] == .5
    assert stats["summary"]["zero_rating_share"] == pytest.approx(2/3)


def test_empty_profile_mapping_has_unknown_coverage_not_one():
    joined, summary = map_user_to_training(normalize_user_lists([]),
        pd.DataFrame({"dataset_anime_id": [1], "myanimelist_id": [431]}),
        pd.DataFrame({"dataset_anime_id": [1]}))
    assert joined.empty
    assert pd.isna(summary["catalog_coverage"])
