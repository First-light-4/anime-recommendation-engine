"""Private in-memory notebook session; source profiles are independent of model version."""

from pathlib import Path

import pandas as pd

from anime_recommender.inference.model_versions import load_version_registry, load_version_workspace, version_franchise_coverage
from anime_recommender.ingestion.yummyanime import YummyAnimeError
from anime_recommender.contracts import DataValidationError
from anime_recommender.profile_service import (
    _public_client, fetch_profile_history, profile_from_history, refresh_profile_recommendations, enrich_profile_titles,
)


class ProfileSession:
    """Cache fetched source history by nickname and derived signals by model/nickname.

    Nothing is persisted. Explicit refresh_profile=True is required to refetch.
    Model/top-K changes never invoke HTTP, fit, or dataset preparation once fetched.
    """

    def __init__(self, root, *, client_factory=None, workspace_loader=load_version_workspace):
        self.root = Path(root)
        self._client_factory = client_factory or (lambda: _public_client(self.root / ".env"))
        self._workspace_loader = workspace_loader
        self._workspaces = {}
        self._sources = {}
        self._profiles = {}
        self.title_cache = {}
        self._titles_attempted = set()

    def workspace(self, model_version):
        if model_version not in self._workspaces:
            self._workspaces[model_version] = self._workspace_loader(self.root, model_version)
        return self._workspaces[model_version]

    def recommend(self, nickname, model_version=None, top_k=15, *, refresh_profile=False):
        if type(top_k) is not int or top_k < 1:
            raise DataValidationError("top_k must be a positive integer")
        if model_version is None:
            model_version = load_version_registry(self.root)["default_version"]
        workspace = self.workspace(model_version)  # Fail before fetching if model is missing.
        nickname = nickname.strip() if nickname else ""
        if not nickname:
            raise DataValidationError("Enter a YummyAnime nickname")
        if refresh_profile:
            self._sources.pop(nickname, None)
            self._profiles = {key: value for key, value in self._profiles.items() if key[1] != nickname}
        if nickname not in self._sources:
            client = self._client_factory()
            if client is None:
                raise DataValidationError("Public application token is not configured")
            try:
                self._sources[nickname] = fetch_profile_history(client, nickname)
            except YummyAnimeError as error:
                self._sources[nickname] = {"error": str(error), "requests": pd.DataFrame(client.request_log)}
        fetched = self._sources[nickname]
        if "error" in fetched:
            return {"state": "error", "error": fetched["error"], "requests": fetched["requests"].copy(),
                    "history": pd.DataFrame(), "recommendations": pd.DataFrame(), "coverage": pd.Series(dtype=object)}
        key = model_version, nickname
        if key not in self._profiles:
            self._profiles[key] = profile_from_history(workspace, fetched, top_k)
            result = dict(self._profiles[key])
        else:
            result = refresh_profile_recommendations(workspace, self._profiles[key], top_k)
        result["recommendations"]["yummyanime_title_ru"] = result["recommendations"].myanimelist_id.map(self.title_cache)
        result["model_version"] = model_version
        result["model_franchise_coverage"] = version_franchise_coverage(workspace, model_version)
        return result

    def remember_titles(self, enriched_profile):
        """Reuse an explicitly fetched title cache, without triggering metadata requests."""
        self.title_cache.update(enriched_profile.get("title_cache", {}))

    def initial_titles(self, profile, nickname):
        """At most one bounded title lookup per nickname/session, never on switching."""
        nickname = nickname.strip()
        if profile["state"] == "ok" and nickname not in self._titles_attempted:
            self._titles_attempted.add(nickname)
            enriched = enrich_profile_titles({**profile, "title_cache": dict(self.title_cache)}, self.root / ".env")
            self.remember_titles(enriched)
            return enriched
        return profile
