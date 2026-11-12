"""Execute notebook analysis offline with synthetic responses, never real profiles."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("pandas")
pytest.importorskip("dotenv")
pytest.importorskip("IPython.display")


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/01_yummyanime_api_exploration.ipynb"


def execute_notebook(monkeypatch, payloads=None):
    monkeypatch.setenv("YUMMYANIME_APPLICATION_TOKEN", "synthetic-token")
    monkeypatch.setenv("YUMMYANIME_TEST_NICKNAME", "synthetic-profile")
    if payloads is None:
        get = Mock(side_effect=requests.ConnectionError("synthetic failure"))
    else:
        responses = []
        for payload in payloads:
            response = Mock(status_code=200)
            response.json.return_value = {"response": payload}
            responses.append(response)
        get = Mock(side_effect=responses)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("IPython.display.display", lambda *args: None)
    namespace = {}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    return namespace, get


def test_null_status_preserved_and_favorites_cross_checked(monkeypatch):
    row = {
        "anime_id": 100,
        "title": "Synthetic title",
        "date": 1700000000,
        "user": {"rating": 0, "list": {"list": None, "is_fav": True}},
        "remote_ids": {"myanimelist_id": 200},
    }
    ns, get = execute_notebook(monkeypatch, [
        {"id": 1, "nickname": "synthetic-profile"}, [row], [row], [],
    ])
    assert get.call_count == 4
    assert ns["snapshot"].rows[0]["user"]["list"]["list"] is None
    assert ns["df"]["list_id"].isna().all()
    assert ns["statuses"].loc["No status", "API"] == 1
    assert ns["snapshot"].favorites["favorite_membership_matches"] is True
    assert ns["comparison"].loc["Favorites", "API"] == 1
    assert ns["comparison"]["Website"].isna().all()
    assert ns["comparison"]["Match"].isna().all()
    assert ns["quality"]["summary"]["duplicate_anime_ids"] == 0
    assert ns["ratings"]["zero_ratings"] == 1


def test_network_failure_stops_requests_and_keeps_comparison_unknown(monkeypatch):
    ns, get = execute_notebook(monkeypatch)
    assert get.call_count == 1
    assert ns["df"].empty
    assert ns["snapshot"].complete is False
    assert ns["comparison"]["Match"].isna().all()
