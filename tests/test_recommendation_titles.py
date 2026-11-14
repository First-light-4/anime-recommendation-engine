"""Source-backed multilingual presentation and bounded metadata requests."""

from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

pd = pytest.importorskip("pandas")

from anime_recommender.inference.recommendation_table import recommendation_table, russian_titles_by_mal
from anime_recommender.ingestion.yummyanime import YummyAnimeClient, YummyAnimeError
from anime_recommender.recommendation_exploration import enrich_profile_titles


def recommendations():
    return pd.DataFrame({"rank": [1, 2, 3], "myanimelist_id": [101, 102, 103],
        "title": ["Primary one", "Primary two", "Primary three"],
        "score": [.7, .5, .3], "model_name": ["als"] * 3})


def test_language_fields_missing_translation_and_no_source_mutation():
    recs = recommendations()
    recs["title_en"] = ["English one", None, " "]
    recs["title_ru"] = ["Русское название", None, ""]
    recs["alternative_title"] = ["Not a Russian translation"] * 3
    recs["yummyanime_title_ru"] = [None, "Название из каталога", None]
    before = recs.copy(deep=True)
    table = recommendation_table(recs)
    assert table["English Title"].tolist() == ["English one", "Primary two", "Primary three"]
    assert table["Russian Title"].tolist() == ["Русское название", "Название из каталога", "—"]
    assert table["ALS Score"].tolist() == [.7, .5, .3]
    pd.testing.assert_frame_equal(recs, before)
    recs["model_name"] = "popularity"
    assert recommendation_table(recs)["ALS Score"].eq("—").all()


def test_exact_mal_unambiguous_primary_title_no_invented_translations():
    rows = [{"remote_ids": {"myanimelist_id": 101}, "title": "Кланнад"},
        {"remote_ids": {"myanimelist_id": 102}, "title": "Hyouka", "other_titles": ["Хёка"]},
        {"remote_ids": {"myanimelist_id": 103}, "title": "Дубликат"},
        {"remote_ids": {"myanimelist_id": 103}, "title": "Другой тайтл"},
        {"remote_ids": {"myanimelist_id": 999}, "title": "Не тот ID"}]
    assert russian_titles_by_mal(rows, [101, 102, 103, 104]) == {101: "Кланнад", 102: None, 103: None, 104: None}


def test_bounded_client_one_filtered_request_and_no_pagination():
    client = YummyAnimeClient("synthetic")
    client._get = Mock(return_value=[])
    assert client.get_anime_by_mal_ids([102, 101]) == []
    client._get.assert_called_once()
    query = parse_qs(urlsplit(client._get.call_args.args[0]).query)
    assert query == {"mal_ids": ["101,102"], "limit": ["20"], "offset": ["0"]}
    for invalid in [[], list(range(1, 22)), [True], [-1]]:
        with pytest.raises(ValueError): client.get_anime_by_mal_ids(invalid)
    assert client._get.call_count == 1


def test_metadata_cache_local_title_precedence_and_ranking_unchanged(monkeypatch):
    client = Mock()
    client.request_log = [{"HTTP": 200}]
    client.get_anime_by_mal_ids.return_value = [{"remote_ids": {"myanimelist_id": 101}, "title": "Первый"}]
    monkeypatch.setattr("anime_recommender.recommendation_exploration._public_client", lambda _: client)
    recs = recommendations()
    recs["title_ru"] = [None, None, "Локальный третий"]
    profile = {"state": "ok", "recommendations": recs}
    first = enrich_profile_titles(profile)
    client.get_anime_by_mal_ids.assert_called_once_with([101, 102])
    second = enrich_profile_titles(first)
    assert client.get_anime_by_mal_ids.call_count == 1
    assert recommendation_table(second["recommendations"])["Russian Title"].tolist() == ["Первый", "—", "Локальный третий"]
    pd.testing.assert_frame_equal(second["recommendations"].drop(columns="yummyanime_title_ru"), recs)


def test_metadata_failure_does_not_change_inference_or_retry(monkeypatch):
    client = Mock()
    client.request_log = [{"HTTP": 403}]
    client.get_anime_by_mal_ids.side_effect = YummyAnimeError("HTTP 403", 403)
    monkeypatch.setattr("anime_recommender.recommendation_exploration._public_client", lambda _: client)
    recs = recommendations()
    first = enrich_profile_titles({"state": "ok", "recommendations": recs})
    second = enrich_profile_titles(first)
    assert second["state"] == "ok" and second["title_error"] == "HTTP 403"
    assert client.get_anime_by_mal_ids.call_count == 1
    assert recommendation_table(second["recommendations"])["Russian Title"].eq("—").all()
