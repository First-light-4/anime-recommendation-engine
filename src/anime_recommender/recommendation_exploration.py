"""In-memory Notebook 04 orchestration, independent of Jupyter and plotting."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from anime_recommender.evaluation.recommendation_checks import (
    diagnostic_metrics, evaluate_validation, generate_validation_recommendations,
)
from anime_recommender.inference.fold_in import fold_in
from anime_recommender.ingestion.training_data import file_sha256
from anime_recommender.ingestion.yummyanime import YummyAnimeClient, YummyAnimeError
from anime_recommender.preprocessing.profile_signals import normalize_profile_history, prepare_profile_signals
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.preprocessing.user_data import STATUS_NAMES
from anime_recommender.training.artifacts import load_models
from anime_recommender.training.inputs import load_training_inputs
from anime_recommender.training.pipeline import run_experiment


@dataclass(repr=False)
class ModelWorkspace:
    bundle: object
    inputs: object
    catalog: pd.DataFrame


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


def annotate_items(frame, workspace):
    """Keep original fields; attach certified titles and train-only popularity."""
    return frame.merge(workspace.catalog[["dataset_anime_id", "title"]], on="dataset_anime_id",
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
    recommenders = workspace.bundle.recommenders()
    als = recommenders["als"].recommend(signals.positive_ids, top_k, seen_items=signals.seen_ids)
    result = als if als.available else recommenders["popularity"].recommend(
        signals.positive_ids, top_k, seen_items=signals.seen_ids)
    signals.frame["status"] = signals.frame.list_id.map(STATUS_NAMES).fillna("No/other status")
    return {"state": "ok", "error": None, "coverage": signals.coverage,
        "history": signals.frame, "recommendations": annotate_items(result.recommendations, workspace),
        "fallback": not als.available, "fallback_reason": als.reason,
        "requests": pd.DataFrame(client.request_log)}


def optional_profile_view(workspace, nickname=None, top_k=10, env_path=None):
    """Opt-in notebook entry point; blank nickname makes zero requests.

    Read the public token without printing it or modifying process environment.
    Import dotenv here so the core inference functions do not require it.
    """
    empty = {"state": "skipped", "error": None, "coverage": pd.Series(dtype=object),
        "history": pd.DataFrame(), "recommendations": pd.DataFrame(), "requests": pd.DataFrame()}
    if not nickname or not nickname.strip():
        return empty
    import os
    from dotenv import dotenv_values
    token = os.environ.get("YUMMYANIME_APPLICATION_TOKEN") or dotenv_values(env_path).get("YUMMYANIME_APPLICATION_TOKEN")
    if not token:
        return {**empty, "state": "error", "error": "Public application token not configured"}
    client = YummyAnimeClient(token)
    try:
        return profile_view(workspace, client, nickname, top_k)
    except YummyAnimeError as error:
        return {**empty, "state": "error", "error": str(error), "requests": pd.DataFrame(client.request_log)}
