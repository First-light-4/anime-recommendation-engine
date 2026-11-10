"""One-profile API probe; never prints response bodies or identifying values.

Run from the project root: python scripts/yummyanime_smoke.py
Configuration is read from the root .env; no additional dependencies required.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener


BASE_URL = "https://api.yani.tv"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class NoRedirect(HTTPRedirectHandler):
    """Do not forward the application token to redirect destinations."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_config(path: Path) -> dict[str, str]:
    """Read the two plain KEY=VALUE settings without logging their contents."""
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError("Invalid .env setting")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def fetch(path: str, label: str, token: str) -> object:
    request = Request(
        BASE_URL + path,
        headers={"X-Application": token, "Accept": "application/json"},
    )
    try:
        # No cookies, owner authorization, retries, or alternate API hosts.
        with build_opener(NoRedirect).open(request, timeout=30) as response:
            print(f"{label}: HTTP {response.status}")
            payload = json.load(response)
    except HTTPError as error:
        print(f"{label}: HTTP {error.code}; stopped (body omitted)")
        raise RuntimeError("API HTTP error") from None
    except (URLError, TimeoutError, OSError):
        print(f"{label}: network error; stopped (details omitted)")
        raise RuntimeError("API connection failed") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("Non-JSON response; stopped") from None
    if not isinstance(payload, dict) or "response" not in payload:
        raise RuntimeError("Unexpected response envelope; stopped")
    if payload.get("error") or payload.get("errors"):
        raise RuntimeError("API reported an error; stopped (details omitted)")
    return payload["response"]


def field(record: dict, *keys: str) -> object:
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def records(value: object) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Unexpected list schema; stopped")
    return value


def main() -> int:
    config = read_config(ENV_PATH)
    token = config.get("YUMMYANIME_APPLICATION_TOKEN", "")
    nickname = config.get("YUMMYANIME_TEST_NICKNAME", "")
    if not token or not nickname:
        print("Not run: set public application token and test nickname in .env.")
        return 2

    profile = fetch("/users/" + quote(nickname, safe=""), "profile", token)
    user_id = field(profile, "id")
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise RuntimeError("Missing integer profile ID; stopped")
    rows = records(fetch(f"/users/{user_id}/lists", "all_lists", token))
    print(f"all_lists: records={len(rows)}")
    paths = {
        "anime_id": ("anime_id",),
        "title": ("title",),
        "user_rating": ("user", "rating"),
        "watch_status": ("user", "list", "list", "id"),
        "favorite": ("user", "list", "is_fav"),
        "myanimelist_id": ("remote_ids", "myanimelist_id"),
        "added_date": ("date",),
    }
    for label, path in paths.items():
        values = [field(row, *path) for row in rows]
        types = Counter(type(value).__name__ for value in values)
        print(f"field {label}: non_null={sum(v is not None for v in values)}; types={dict(types)}")

    # Check all six documented lists of this one profile, including favorites.
    # This detects discrepancies, but cannot prove absence of a shared server cap.
    all_ids = {field(row, "anime_id") for row in rows}
    statuses = {field(row, *paths["watch_status"]) for row in rows}
    valid_statuses = {0, 1, 2, 3, 4, 5}
    if not statuses <= valid_statuses | {None}:
        raise RuntimeError("Unknown or missing list status; completeness unverified")
    print(f"records_without_watch_status={sum(field(row, *paths['watch_status']) is None for row in rows)}")
    partition_ids = set()
    for status in sorted(valid_statuses):
        subset = records(fetch(f"/users/{user_id}/lists/{status}", f"list_{status}", token))
        expected = {
            field(row, "anime_id") for row in rows
            if (field(row, "user", "list", "is_fav") is True if status == 4
                else field(row, *paths["watch_status"]) == status)
        }
        actual = {field(row, "anime_id") for row in subset}
        print(f"list_{status}: records={len(subset)}; matches_all_lists={actual == expected}")
        partition_ids.update(actual)
    print(f"completeness: union_matches={partition_ids == all_ids}; server_cap_not_excluded=True")
    print("User rating ownership and status-change history require separate verification.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as error:
        # Exception values from libraries may contain URLs or credentials.
        print(f"Stopped: {type(error).__name__}; no response data saved.")
        sys.exit(1)
