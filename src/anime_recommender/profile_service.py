"""Profile application service: explicit source access and saved-model inference.
No training, evaluation or research imports. Source history remains memory-only.
"""
import pandas as pd
from anime_recommender.contracts import DataValidationError
from anime_recommender.ingestion.yummyanime import YummyAnimeClient, YummyAnimeError
from anime_recommender.preprocessing.profile_signals import normalize_profile_history, prepare_profile_signals
from anime_recommender.preprocessing.user_data import STATUS_NAMES
from anime_recommender.inference.franchise_filter import recommend_with_franchise
from anime_recommender.inference.recommendation_table import recommendation_table, russian_titles_by_mal
from anime_recommender.inference.presentation import annotate_items

def profile_view(workspace, client, nickname, top_k=10):
    """Three bounded public API requests, then in-memory inference only.

    API failure propagates to caller without retry, fallback or partial inference.
    Popularity fallback is used only for a successfully fetched empty warm context.
    """
    fetched = fetch_profile_history(client, nickname)
    return profile_from_history(workspace, fetched, top_k)

def fetch_profile_history(client, nickname):
    """Fetch three bounded endpoints, retaining only normalized source data in memory."""
    profile = client.get_user(nickname)
    rows = client.get_lists(profile["id"])
    favorites = client.get_favorites(profile["id"])
    history = normalize_profile_history(rows, favorites)
    return {"history": history, "requests": pd.DataFrame(client.request_log)}

def profile_from_history(workspace, fetched_history, top_k=10):
    """Recompute mapping against the selected vocabulary, without changing source history."""
    history = fetched_history["history"].copy(deep=True)
    signals = prepare_profile_signals(history, workspace.catalog, workspace.bundle.items)
    signals.frame["status"] = signals.frame.list_id.map(STATUS_NAMES).fillna("No/other status")
    fetched = {"state": "ok", "error": None, "coverage": signals.coverage,
        "history": signals.frame, "signals": signals, "requests": fetched_history["requests"].copy()}
    return refresh_profile_recommendations(workspace, fetched, top_k)

def refresh_profile_recommendations(workspace, profile, top_k=10):
    """Rerank an already fetched profile, with zero API calls and no training."""
    if profile["state"] != "ok":
        return profile
    signals = profile["signals"]
    recommenders = workspace.bundle.recommenders()
    config = workspace.franchise_config
    if config is not None and config["enabled"]:
        history = profile["history"]
        observed = history.myanimelist_id.dropna().astype(int).tolist()
        completed = history.loc[history.list_id.eq(2).fillna(False), "myanimelist_id"].dropna().astype(int).tolist()
        kwargs = {"train_items": workspace.bundle.items, "positive_context": signals.positive_ids,
            "mapping": workspace.franchise_mapping, "observed_dataset_ids": signals.seen_ids,
            "observed_mal_ids": observed, "completed_mal_ids": completed, "top_k": top_k,
            "max_per_franchise": config["max_recommendations_per_franchise"],
            "candidate_multiplier": config["candidate_multiplier"]}
        filtered = recommend_with_franchise(recommenders["als"], **kwargs)
        fallback = not filtered.diagnostics["model_available"]
        unavailable_reason = filtered.diagnostics["model_unavailable_reason"]
        if fallback:
            filtered = recommend_with_franchise(recommenders["popularity"], **kwargs)
        filtered.diagnostics.update({"metadata_warning": config.get("metadata_warning"),
            "metadata_source": config.get("metadata_source", "configured mapping")})
        return {**profile, "recommendations": annotate_items(filtered.recommendations, workspace),
            "fallback": fallback, "fallback_reason": unavailable_reason,
            "franchise_diagnostics": filtered.diagnostics, "franchise_audit": filtered.audit}
    als = recommenders["als"].recommend(signals.positive_ids, top_k, seen_items=signals.seen_ids)
    result = als if als.available else recommenders["popularity"].recommend(
        signals.positive_ids, top_k, seen_items=signals.seen_ids)
    return {**profile, "recommendations": annotate_items(result.recommendations, workspace),
        "fallback": not als.available, "fallback_reason": als.reason,
        "franchise_diagnostics": {"enabled": False}, "franchise_audit": pd.DataFrame()}

def optional_profile_view(workspace, nickname=None, top_k=10, env_path=None, *, _client_factory=None):
    """Opt-in notebook entry point; blank nickname makes zero requests.

    Read the public token without printing it or modifying process environment.
    Import dotenv here so the core inference functions do not require it.
    """
    empty = {"state": "skipped", "error": None, "coverage": pd.Series(dtype=object),
        "history": pd.DataFrame(), "recommendations": pd.DataFrame(), "requests": pd.DataFrame()}
    if not nickname or not nickname.strip():
        return empty
    client = (_client_factory or _public_client)(env_path)
    if client is None:
        return {**empty, "state": "error", "error": "Public application token not configured"}
    try:
        return profile_view(workspace, client, nickname, top_k)
    except YummyAnimeError as error:
        message = "Profile not found (HTTP 404); check the nickname and profile availability." if error.status == 404 else str(error)
        return {**empty, "state": "error", "error": message, "requests": pd.DataFrame(client.request_log)}

def _public_client(env_path):
    import os
    from dotenv import dotenv_values
    token = os.environ.get("YUMMYANIME_APPLICATION_TOKEN") or dotenv_values(env_path).get("YUMMYANIME_APPLICATION_TOKEN")
    return YummyAnimeClient(token) if token else None

def enrich_profile_titles(profile, env_path=None, *, _client_factory=None):
    """At most one metadata call for up to 20 selected MAL IDs; memory cache only.

    Metadata failure never changes recommendation IDs, scores or fallback choice.
    Missing/ambiguous/failed titles remain unavailable, with no automatic retries.
    """
    if profile["state"] != "ok":
        return profile
    cache = dict(profile.get("title_cache", {}))
    recs = profile["recommendations"].copy()
    needs_title = recommendation_table(recs)["Russian Title"].eq("—").to_numpy()
    ids = [int(value) for value in recs.loc[needs_title, "myanimelist_id"].dropna().unique()]
    pending = [value for value in ids if value not in cache][:20]
    error = profile.get("title_error")
    log = profile.get("title_requests", pd.DataFrame())
    if pending:
        cache.update(dict.fromkeys(pending))
        client = (_client_factory or _public_client)(env_path)
        if client is None:
            error = "Public application token not configured; Russian titles unavailable"
        else:
            try:
                cache.update(russian_titles_by_mal(client.get_anime_by_mal_ids(pending), pending))
                error = None
            except YummyAnimeError as failure:
                error = str(failure)
            new_log = pd.DataFrame(client.request_log)
            log = new_log if log.empty else pd.concat([log, new_log], ignore_index=True)
    recs["yummyanime_title_ru"] = recs.myanimelist_id.map(cache)
    return {**profile, "recommendations": recs, "title_cache": cache,
            "title_error": error, "title_requests": log}
