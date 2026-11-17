"""Read-only YummyAnime client. Public application token only; no Jupyter state."""

from urllib.parse import quote, urlencode

import requests


class YummyAnimeError(RuntimeError):
    """Sanitized error: never includes token, nickname, URL, or response body."""

    def __init__(self, message: str, status: int | None = None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class YummyAnimeAuthorizationError(YummyAnimeError):
    pass


class YummyAnimeNetworkError(YummyAnimeError):
    pass


class YummyAnimeClient:
    """At most one HTTP attempt per call, no redirects or owner authorization."""

    BASE_URL = "https://api.yani.tv"

    def __init__(self, application_token: str, timeout: tuple[float, float] = (10, 30)):
        if not isinstance(application_token, str) or not application_token.strip():
            raise ValueError("Public application token is required")
        self._token = application_token.strip()
        self.timeout = timeout
        self.request_log: list[dict] = []

    def _get(self, path: str, label: str):
        try:
            response = requests.get(self.BASE_URL + path,
                headers={"X-Application": self._token, "Accept": "application/json"},
                timeout=self.timeout, allow_redirects=False)
        except requests.RequestException:
            self.request_log.append({"endpoint": label, "HTTP": None, "result": "network error"})
            raise YummyAnimeNetworkError("API network/timeout error; details omitted") from None
        status = response.status_code
        self.request_log.append({"endpoint": label, "HTTP": status,
                                 "result": "OK" if status == 200 else "stopped"})
        if status in (401, 403):
            raise YummyAnimeAuthorizationError(f"API authorization error: HTTP {status}", status)
        if status != 200:
            raise YummyAnimeError(f"API request failed: HTTP {status}", status,
                                 retry_after=response.headers.get("Retry-After") if status in (429, 503) else None)
        try:
            payload = response.json()
        except ValueError:
            raise YummyAnimeError("API returned invalid JSON") from None
        if (not isinstance(payload, dict) or "response" not in payload
                or payload.get("error") or payload.get("errors")):
            raise YummyAnimeError("Unexpected API envelope or API error; body omitted")
        return payload["response"]

    @staticmethod
    def _user_id(user_id: int) -> int:
        if type(user_id) is not int or user_id <= 0:
            raise ValueError("user_id must be a positive integer")
        return user_id

    @staticmethod
    def _records(payload) -> list[dict]:
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise YummyAnimeError("Unexpected API list schema; body omitted")
        return payload

    def get_user(self, nickname: str) -> dict:
        if not isinstance(nickname, str) or not nickname.strip():
            raise ValueError("nickname must be nonempty")
        profile = self._get("/users/" + quote(nickname, safe=""), "GET /users/{nickname}")
        if not isinstance(profile, dict) or type(profile.get("id")) is not int or profile["id"] <= 0:
            raise YummyAnimeError("Profile response has no positive integer ID")
        return profile

    def get_lists(self, user_id: int) -> list[dict]:
        return self._records(self._get(f"/users/{self._user_id(user_id)}/lists", "GET /users/{id}/lists"))

    def get_favorites(self, user_id: int) -> list[dict]:
        return self._records(self._get(f"/users/{self._user_id(user_id)}/lists/4", "GET /users/{id}/lists/4"))

    def get_rating_statistics(self, user_id: int) -> list[dict]:
        return self._records(self._get(f"/users/{self._user_id(user_id)}/stats/ratings", "GET /users/{id}/stats/ratings"))

    def get_anime_by_mal_ids(self, mal_ids: list[int]) -> list[dict]:
        """One bounded catalog request, no pagination or automatic retries."""
        return self._catalog_by_ids("mal_ids", mal_ids)

    def get_anime_by_ids(self, anime_ids: list[int]) -> list[dict]:
        return self._catalog_by_ids("ids", anime_ids)

    def _catalog_by_ids(self, field, values):
        if not values or len(values) > 20 or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Request between 1 and 20 positive integer IDs")
        ids = sorted(set(values))
        query = urlencode({field: ",".join(map(str, ids)), "limit": 20, "offset": 0})
        label = "GET /anime (bounded MAL filter)" if field == "mal_ids" else "GET /anime (bounded anime-ID filter)"
        rows = self._records(self._get("/anime?" + query, label))
        if len(rows) > 20:
            raise YummyAnimeError("Unexpected catalog result size; metadata ignored")
        return rows

    def get_anime(self, anime_ref):
        if not isinstance(anime_ref, (str, int)) or isinstance(anime_ref, bool) or not str(anime_ref).strip() or (type(anime_ref) is int and anime_ref <= 0):
            raise ValueError("Anime ID or alias required")
        result = self._get("/anime/" + quote(str(anime_ref), safe=""), "GET /anime/{url}")
        if not isinstance(result, dict) or type(result.get("anime_id")) is not int or result["anime_id"] <= 0:
            raise YummyAnimeError("Unexpected anime detail schema")
        return result

    def get_anime_collections(self, anime_id, limit=1):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Collections research limit must be 1..20")
        return self._records(self._get(f"/anime/{self._user_id(anime_id)}/collections?limit={limit}",
                                       "GET /anime/{id}/collections"))
