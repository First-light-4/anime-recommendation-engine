"""In-memory Notebook 04 orchestration, independent of Jupyter and plotting."""

import json
from pathlib import Path

import pandas as pd

from anime_recommender.inference.fold_in import fold_in
from anime_recommender.artifact_files import file_sha256
from anime_recommender.ingestion.yummyanime import YummyAnimeClient
from anime_recommender.contracts import DataValidationError
from anime_recommender.artifact_io import load_models


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


# Compatibility adapters retained through ARE-009D/E. New serving code does not
# import this research facade. Lazy research calls are never used by model load.
from anime_recommender.contracts import ModelWorkspace
from anime_recommender.inference.presentation import annotate_items, profile_display
from anime_recommender.profile_service import profile_view, fetch_profile_history, profile_from_history, refresh_profile_recommendations
from anime_recommender import profile_service as _profiles
from anime_recommender.registry.workspace import load_profile_workspace as _load_profile_workspace

def load_profile_workspace(root, model_dir=None):
    return _load_profile_workspace(root, model_dir, model_loader=load_models)

def _public_client(env_path):
    import os
    from dotenv import dotenv_values
    token = os.environ.get("YUMMYANIME_APPLICATION_TOKEN") or dotenv_values(env_path).get("YUMMYANIME_APPLICATION_TOKEN")
    return YummyAnimeClient(token) if token else None

def optional_profile_view(workspace, nickname=None, top_k=10, env_path=None):
    return _profiles.optional_profile_view(workspace, nickname, top_k, env_path, _client_factory=_public_client)

def enrich_profile_titles(profile, env_path=None):
    return _profiles.enrich_profile_titles(profile, env_path, _client_factory=_public_client)

def load_training_inputs(*args, **kwargs):
    from anime_recommender.training.inputs import load_training_inputs as call
    return call(*args, **kwargs)

def run_experiment(*args, **kwargs):
    from anime_recommender.training.pipeline import run_experiment as call
    return call(*args, **kwargs)

def evaluate_validation(*args, **kwargs):
    from anime_recommender.evaluation.recommendation_checks import evaluate_validation as call
    return call(*args, **kwargs)

def diagnostic_metrics(*args, **kwargs):
    from anime_recommender.evaluation.recommendation_checks import diagnostic_metrics as call
    return call(*args, **kwargs)

def generate_validation_recommendations(*args, **kwargs):
    from anime_recommender.evaluation.recommendation_checks import generate_validation_recommendations as call
    return call(*args, **kwargs)
