"""ARE-006: fit on train, persist models, diagnose on validation only."""

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from anime_recommender.evaluation.recommendation_checks import evaluate_validation
from anime_recommender.inference.fold_in import fold_in
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.training.als import train_als
from anime_recommender.training.artifacts import ModelBundle, load_models, save_models
from anime_recommender.training.inputs import load_training_inputs
from anime_recommender.training.popularity import train_popularity


@dataclass
class Experiment:
    bundle: ModelBundle
    diagnostics: dict
    examples: dict[str, pd.DataFrame]
    score_summary: pd.Series
    example_user_id: int


def validate_config(config, root):
    if set(config) != {"dataset_dir", "catalog_path", "model_dir", "diagnostic_dir", "top_k", "als", "blas_threads"}:
        raise DataValidationError("Unexpected training pipeline configuration")
    if type(config["top_k"]) is not int or config["top_k"] < 1:
        raise DataValidationError("top_k must be positive")
    for key in ("dataset_dir", "catalog_path", "model_dir", "diagnostic_dir"):
        path = Path(config[key])
        parent = root / ("data" if key in ("dataset_dir", "catalog_path") else "artifacts")
        if path.is_absolute() or not (root / path).resolve().is_relative_to(parent.resolve()):
            raise DataValidationError(f"{key} must be relative and inside {parent.name}/")
    if config["model_dir"] == config["diagnostic_dir"]:
        raise DataValidationError("Separate models from diagnostics")


def fit_models(inputs, config):
    start = perf_counter()
    popularity = train_popularity(inputs.train, inputs.items)
    pop_seconds = perf_counter() - start
    start = perf_counter()
    als = train_als(inputs.matrix, config["als"], config["blas_threads"])
    als_seconds = perf_counter() - start
    metadata = {"format_version": 1, "training_config": config,
        "dataset_config": inputs.dataset_config, "train_fingerprint": inputs.train_fingerprint,
        "train_hashes": inputs.train_hashes, "matrix_orientation": "users_by_items",
        "train_shape": list(inputs.matrix.shape), "train_nnz": inputs.matrix.nnz,
        "fit_seconds": {"popularity": pop_seconds, "als": als_seconds},
        "factor_shapes": {"users": list(als.user_factors.shape), "items": list(als.item_factors.shape)},
        "versions": {name: version(name) for name in ("implicit", "numpy", "scipy", "pandas", "pyarrow", "threadpoolctl")},
        "confidence_policy": "binary confidence=1, implicit alpha=1; no tuning",
        "training_split": "train only"}
    return ModelBundle(als, popularity, inputs.items.copy(), inputs.train_users.copy(), metadata)


def _compare_reload(before, after, inputs, top_k):
    left, right = before.recommenders(), after.recommenders()
    contexts = {int(uid): group.dataset_anime_id.tolist() for uid, group in inputs.validation_context.groupby("dataset_user_id")}
    seen = {int(uid): group.dataset_anime_id.tolist() for uid, group in inputs.observed_context.groupby("dataset_user_id")}
    for uid in inputs.validation_users.dataset_user_id:
        for name in left:
            a = left[name].recommend(contexts.get(int(uid), []), top_k, seen_items=seen.get(int(uid), []))
            b = right[name].recommend(contexts.get(int(uid), []), top_k, seen_items=seen.get(int(uid), []))
            if (a.available != b.available or a.recommendations.dataset_anime_id.tolist() != b.recommendations.dataset_anime_id.tolist()
                    or not np.allclose(a.recommendations.score.to_numpy(dtype=float), b.recommendations.score.to_numpy(dtype=float), rtol=1e-5, atol=1e-6)):
                raise DataValidationError("Save/load changed validation recommendations")


def run_experiment(config, root, train=True) -> Experiment:
    root = Path(root).resolve()
    validate_config(config, root)
    inputs = load_training_inputs(root / config["dataset_dir"])
    if train:
        original = fit_models(inputs, config)
        save_models(original, root / config["model_dir"])
        bundle = load_models(root / config["model_dir"], inputs.items, inputs.train_fingerprint)
        _compare_reload(original, bundle, inputs, config["top_k"])
        bundle.metadata["serialization_checked_validation_users"] = len(inputs.validation_users)
        atomic_json(root / config["model_dir"] / "metadata.json", bundle.metadata)
    else:
        bundle = load_models(root / config["model_dir"], inputs.items, inputs.train_fingerprint)
        if bundle.metadata["training_config"]["als"] != config["als"]:
            raise DataValidationError("Saved model config differs; train explicitly instead of silently reusing")
    before_items = bundle.als.item_factors.copy()
    before_users = bundle.als.user_factors.copy()
    diagnostics = evaluate_validation(bundle.recommenders(), inputs.validation_context,
        inputs.observed_context, inputs.validation_target, inputs.validation_users, inputs.items, config["top_k"])
    if not np.array_equal(before_items, bundle.als.item_factors) or not np.array_equal(before_users, bundle.als.user_factors):
        raise DataValidationError("Validation fold-in changed trained user/item factors")
    if file_sha256(root / config["catalog_path"]) != inputs.catalog_sha256:
        raise DataValidationError("Display catalog differs from model dataset catalog")
    catalog = pd.read_parquet(root / config["catalog_path"])[["dataset_anime_id", "title"]]
    uid = int(inputs.validation_users.dataset_user_id.iloc[0])
    if uid in set(inputs.train_users.dataset_user_id):
        raise DataValidationError("Example user must be absent from train")
    examples = {}
    for name in bundle.recommenders():
        example = diagnostics["recommendations"].loc[lambda frame: frame.dataset_user_id.eq(uid) & frame.model_name.eq(name)]
        examples[name] = example.merge(catalog, on="dataset_anime_id", how="left", validate="many_to_one")
        if examples[name].title.isna().any():
            raise DataValidationError("Recommendation title missing from certified catalog")
    context_ids = inputs.validation_context.loc[inputs.validation_context.dataset_user_id.eq(uid), "dataset_anime_id"].tolist()
    folded = fold_in(bundle.als, context_ids, bundle.items)
    scores = pd.Series(bundle.als.item_factors @ folded.factor, name="ALS score (not probability)") if folded.available else pd.Series([], dtype=float)
    score_summary = scores.describe() if len(scores) else pd.Series({"count": 0})
    directory = root / config["diagnostic_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in diagnostics.items():
        atomic_parquet(frame, directory / (name + ".parquet"))
    for name, frame in examples.items():
        atomic_parquet(frame, directory / (name + "_example.parquet"))
    report = {"split": "validation", "top_k": config["top_k"],
        "metrics": diagnostics["summary"].to_dict("records"), "example_user_absent_from_train": True,
        "same_candidates": "train vocabulary minus all observed context items, including low/zero ratings",
        "recall_definition": "Recall@K aliases Full Recall; NDCG ideal denominator includes cold targets",
        "warm_definition": "warm context AND nonempty warm target; conditional diagnostic, not whole service",
        "factor_immutability": True,
        "save_load_checked_on_training_run_users": bundle.metadata.get("serialization_checked_validation_users", 0),
        "score_summary": score_summary.to_dict(), "test_evaluated": False}
    atomic_json(directory / "report.json", report)
    return Experiment(bundle, diagnostics, examples, score_summary, uid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/recommendation_models.json")
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    result = run_experiment(config, root, train=not args.load_only)
    print(json.dumps({"fit_seconds": result.bundle.metadata["fit_seconds"],
        "factor_shapes": result.bundle.metadata["factor_shapes"]}, indent=2))
    print(result.diagnostics["summary"].to_string(index=False))
    for name, frame in result.examples.items():
        print(name)
        print(frame[["rank", "title", "dataset_anime_id", "myanimelist_id", "score"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ARE-006 failed: {error}") from None
