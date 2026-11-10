"""Offline checks for probe privacy and stop-on-error behavior."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts" / "yummyanime_smoke.py"
    spec = importlib.util.spec_from_file_location("yummyanime_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_config_makes_no_requests(probe, tmp_path, monkeypatch):
    config = tmp_path / ".env"
    config.write_text("YUMMYANIME_APPLICATION_TOKEN=\nYUMMYANIME_TEST_NICKNAME=\n")
    monkeypatch.setattr(probe, "ENV_PATH", config)
    fetch = MagicMock()
    monkeypatch.setattr(probe, "fetch", fetch)

    assert probe.main() == 2
    fetch.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_failure_stops_without_disclosing_data(probe, monkeypatch, capsys, status):
    opener = MagicMock()
    opener.open.side_effect = HTTPError(
        "https://example.invalid/private-nickname", status, "sensitive-body", {}, None
    )
    monkeypatch.setattr(probe, "build_opener", lambda *args: opener)

    with pytest.raises(RuntimeError):
        probe.fetch("/users/private-nickname", "profile", "secret-token")

    output = capsys.readouterr().out
    assert f"HTTP {status}" in output
    assert "secret-token" not in output
    assert "private-nickname" not in output
    assert "sensitive-body" not in output
    assert opener.open.call_count == 1


def test_redirect_is_not_followed(probe):
    assert probe.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid") is None


def test_favorites_without_status_are_included_in_completeness(probe, monkeypatch, capsys):
    row = {"anime_id": 10, "user": {"list": {"list": None, "is_fav": True}}}
    monkeypatch.setattr(probe, "read_config", lambda path: {
        "YUMMYANIME_APPLICATION_TOKEN": "test-token",
        "YUMMYANIME_TEST_NICKNAME": "test-profile",
    })
    fetch = MagicMock(side_effect=[
        {"id": 1}, [row], [], [], [], [], [row], [],
    ])
    monkeypatch.setattr(probe, "fetch", fetch)

    assert probe.main() == 0
    output = capsys.readouterr().out
    assert "records_without_watch_status=1" in output
    assert "list_4: records=1; matches_all_lists=True" in output
    assert "union_matches=True" in output
    assert fetch.call_count == 8
