"""In-memory Notebook 04 orchestration, independent of Jupyter and plotting."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from anime_recommender.evaluation.recommendation_checks import (
    diagnostic_metrics, evaluate_validation, generate_validation_recommendations,
)
from anime_recommender.inference.fold_in import fold_in
from anime_recommender.inference.franchise_filter import recommend_with_franchise
from anime_recommender.inference.recommendation_table import (
    ENGLISH_FIELDS, RUSSIAN_FIELDS, recommendation_table, russian_titles_by_mal,
)
from anime_recommender.ingestion.training_data import file_sha256
from anime_recommender.ingestion.yummyanime import YummyAnimeClient, YummyAnimeError
from anime_recommender.preprocessing.profile_signals import normalize_profile_history, prepare_profile_signals
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.preprocessing.user_data import STATUS_NAMES
from anime_recommender.preprocessing.franchise_mapping import load_franchise_resources
from anime_recommender.training.artifacts import load_models
from anime_recommender.training.inputs import load_training_inputs
from anime_recommender.training.pipeline import run_experiment


@dataclass(repr=False)
class ModelWorkspace:
    bundle: object
    inputs: object
    catalog: pd.DataFrame
    franchise_config: dict | None = None
    franchise_mapping: object | None = None


def load_workspace(root, factors=64, regularization=0.1, iterations=20, seed=42, retrain=False):
    """Model parameter mismatch raises until explicit retrain=True; no silent fitting.

    Nondefault configurations are saved separately from the ARE-006 baseline.
    top_k and selected user are deliberately not model-loading parameters.
    """
    root = Path(root)
    config = json.loads((root / "configs/recommendation_models.json").read_text(encoding="utf-8"))
    original = config["als"].copy()
    config["als"].update(factors=factors, regularization=regularization, iterations=iterations, random_state=seed)
    if config["als"] != original:
        config["model_dir"] += "-notebook"
        config["diagnostic_dir"] += "-notebook"
    inputs = load_training_inputs(root / config["dataset_dir"])
    if retrain:
        bundle = run_experiment(config, root, train=True).bundle
    else:
        try:
            bundle = load_models(root / config["model_dir"], inputs.items, inputs.train_fingerprint)
        except FileNotFoundError:
            raise DataValidationError("No saved model for these parameters; set retrain=True explicitly") from None
        if bundle.metadata["training_config"]["als"] != config["als"]:
            raise DataValidationError("ALS parameters changed; set retrain=True explicitly")
    path = root / config["catalog_path"]
    if file_sha256(path) != inputs.catalog_sha256:
        raise DataValidationError("Display catalog differs from certified dataset")
    return ModelWorkspace(bundle, inputs, pd.read_parquet(path))


def load_profile_workspace(root):
    """Load only saved models and the certified catalog, with no training/splits.

    A missing bundle never invokes fit or downloads anything. The preparation
    instruction deliberately points to the separate existing training pipeline.
    """
    root = Path(root)
    config = json.loads((root / "configs/recommendation_models.json").read_text(encoding="utf-8"))
    try:
        bundle = load_models(root / config["model_dir"])
        path = root / config["catalog_path"]
        report = json.loads((path.parent / "data_quality_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "success" or file_sha256(path) != report.get("output_sha256", {}).get(path.name):
            raise DataValidationError("Display catalog is not certified by ARE-004")
        catalog = pd.read_parquet(path)
        aligned = bundle.items[["dataset_anime_id", "myanimelist_id"]].merge(
            catalog[["dataset_anime_id", "myanimelist_id"]], on="dataset_anime_id",
            how="left", validate="one_to_one", suffixes=("_model", "_catalog"))
        if not aligned.myanimelist_id_model.eq(aligned.myanimelist_id_catalog).all():
            raise DataValidationError("Catalog MAL mapping differs from saved model")
    except FileNotFoundError:
        raise DataValidationError(
            "Saved models/catalog are missing. Prepare local ARE-005.1 data, then run "
            "python -m anime_recommender.training.pipeline from the project root. "
            "Notebook 05 only loads models and never trains them.") from None
    franchise_config, franchise_mapping = load_franchise_resources(root)
    return ModelWorkspace(bundle, None, catalog, franchise_config, franchise_mapping)


def annotate_items(frame, workspace):
    """Keep original fields; attach certified titles and train-only popularity."""
    titles = list(dict.fromkeys(["title", *ENGLISH_FIELDS, *RUSSIAN_FIELDS]))
    columns = ["dataset_anime_id", *[name for name in titles if name in workspace.catalog]]
    return frame.merge(workspace.catalog[columns], on="dataset_anime_id",
                       how="left", validate="many_to_one").merge(
        workspace.bundle.popularity[["dataset_anime_id", "unique_users"]], on="dataset_anime_id",
        how="left", validate="many_to_one")


def score_distribution(workspace, positive_ids):
    folded = fold_in(workspace.bundle.als, positive_ids, workspace.bundle.items)
    scores = workspace.bundle.als.item_factors @ folded.factor if folded.available else []
    return pd.Series(scores, dtype=float, name="ALS score (not probability)")


def validation_diagnostics(workspace, top_k=10):
    data = workspace.inputs
    return evaluate_validation(workspace.bundle.recommenders(), data.validation_context,
        data.observed_context, data.validation_target, data.validation_users, workspace.bundle.items, top_k)


def validation_view(workspace, user_id=None, top_k=10):
    data = workspace.inputs
    users = data.validation_users
    user_id = int(users.dataset_user_id.iloc[0]) if user_id is None else user_id
    if user_id not in set(users.dataset_user_id):
        raise DataValidationError("Select a user_id from validation_user_mapping")
    selected = users.loc[users.dataset_user_id.eq(user_id)]
    context = data.validation_context.loc[data.validation_context.dataset_user_id.eq(user_id)]
    observed = data.observed_context.loc[data.observed_context.dataset_user_id.eq(user_id)]
    # Generation never receives targets. All annotations/metrics happen afterwards.
    recs, statuses = generate_validation_recommendations(workspace.bundle.recommenders(), selected,
        context, observed, workspace.bundle.items, top_k)
    target = data.validation_target.loc[data.validation_target.dataset_user_id.eq(user_id)]
    summary, per_user = diagnostic_metrics(recs, statuses, target, selected, workspace.bundle.items, top_k)
    recommendations = annotate_items(recs, workspace)
    recommendations["is_hit"] = recommendations.dataset_anime_id.isin(target.dataset_anime_id)
    history = annotate_items(observed, workspace)
    history["is_warm"] = history.dataset_anime_id.isin(workspace.bundle.items.dataset_anime_id)
    history["is_positive"] = history.dataset_anime_id.isin(context.dataset_anime_id)
    targets = annotate_items(target, workspace)
    targets["is_warm"] = targets.dataset_anime_id.isin(workspace.bundle.items.dataset_anime_id)
    hits = recommendations.groupby("model_name").is_hit.sum().reindex(
        ["popularity", "als"], fill_value=0).to_frame("target_hits")
    return {"user_id": user_id, "context": history, "positive_targets": targets,
        "recommendations": recommendations, "metrics": summary, "per_user": per_user,
        "statuses": statuses, "scores": score_distribution(workspace, context.dataset_anime_id.tolist()),
        "hits": hits}


def profile_view(workspace, client, nickname, top_k=10):
    """Three bounded public API requests, then in-memory inference only.

    API failure propagates to caller without retry, fallback or partial inference.
    Popularity fallback is used only for a successfully fetched empty warm context.
    """
    profile = client.get_user(nickname)
    rows = client.get_lists(profile["id"])
    favorites = client.get_favorites(profile["id"])
    history = normalize_profile_history(rows, favorites)
    signals = prepare_profile_signals(history, workspace.catalog, workspace.bundle.items)
    signals.frame["status"] = signals.frame.list_id.map(STATUS_NAMES).fillna("No/other status")
    fetched = {"state": "ok", "error": None, "coverage": signals.coverage,
        "history": signals.frame, "signals": signals, "requests": pd.DataFrame(client.request_log)}
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


def optional_profile_view(workspace, nickname=None, top_k=10, env_path=None):
    """Opt-in notebook entry point; blank nickname makes zero requests.

    Read the public token without printing it or modifying process environment.
    Import dotenv here so the core inference functions do not require it.
    """
    empty = {"state": "skipped", "error": None, "coverage": pd.Series(dtype=object),
        "history": pd.DataFrame(), "recommendations": pd.DataFrame(), "requests": pd.DataFrame()}
    if not nickname or not nickname.strip():
        return empty
    client = _public_client(env_path)
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


def enrich_profile_titles(profile, env_path=None):
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
        client = _public_client(env_path)
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


def profile_display(profile, nickname, show_history=False):
    """Presentation-only data, no HTTP, signal preparation or ranking duplication.

    No public YummyAnime catalog mapping/URLs are available for recommended items;
    never construct a YummyAnime URL from an unrelated MAL or dataset ID.
    """
    if profile["state"] != "ok":
        return {"summary": pd.Series({"State": profile["state"], "Error": profile["error"]}),
                "recommendations": pd.DataFrame(), "history": pd.DataFrame()}
    history = profile["history"]
    coverage = profile["coverage"]
    recs = profile["recommendations"]
    used = "popularity" if profile["fallback"] else "als"
    summary = pd.Series({
        "Nickname": nickname,
        "Anime total": history.anime_id.nunique(),
        "Completed": history.loc[history.list_id.eq(2), "anime_id"].nunique(),
        "Favorites": history.loc[history.is_fav.fillna(False), "anime_id"].nunique(),
        "Rated": history.loc[history.user_rating.between(1, 10).fillna(False), "anime_id"].nunique(),
        "Positive interactions": history.loc[history.positive, "anime_id"].nunique(),
        "Warm positives for fold-in": coverage["fold_in_interactions"],
        "MAL coverage (catalog)": coverage["catalog_coverage"],
        "MAL coverage (train vocabulary)": coverage["training_coverage"],
        "Matched train items": coverage["training_matches"],
        "Excluded candidate items": history.loc[history.known_in_training, "dataset_anime_id"].nunique(),
        "Model": used, "Fallback reason": profile["fallback_reason"],
    })
    table = recommendation_table(recs)
    return {"summary": summary, "recommendations": table,
            "history": history if show_history else history.iloc[:0].copy()}
