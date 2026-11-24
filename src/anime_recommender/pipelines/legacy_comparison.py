"""ARE-008B: identical ALS settings, fixed validation, no franchise filtering/test."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from anime_recommender.evaluation.scaling_overview import (
    fixed_validation_identity, check_comparison_compatibility, load_comparison, selective_scaling_overview,
)
from anime_recommender.evaluation.ranking_evaluator import evaluate_models, evaluation_fingerprint, model_fingerprint
from anime_recommender.evaluation.reports import save_report
from anime_recommender.inference.model_versions import load_version_registry
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.recommendation_exploration import ModelWorkspace, validation_view
from anime_recommender.training.artifacts import load_models, save_models
from anime_recommender.training.inputs import load_training_inputs
from anime_recommender.training.fitting import fit_models


def comparison_row(version, inputs, bundle, report, reused):
    metrics, coverage = report.tables["metrics"], report.tables["coverage"]
    def metric(model, mode, field):
        return float(metrics.loc[metrics.model_name.eq(model) & metrics["mode"].eq(mode) & metrics.k.eq(10), field].iloc[0])
    return {"model_version": version, "train_users": len(inputs.train_users), "train_items": len(inputs.items),
        "train_positives": len(inputs.train), "als_full_recall_10": metric("als", "full", "recall"),
        "als_full_ndcg_10": metric("als", "full", "ndcg"), "als_warm_recall_10": metric("als", "warm", "recall"),
        "popularity_recall_10": metric("popularity", "full", "recall"),
        "popularity_ndcg_10": metric("popularity", "full", "ndcg"),
        "catalog_coverage_10": float(coverage.loc[coverage.model_name.eq("als") & coverage.k.eq(10), "catalog_coverage"].iloc[0]),
        "popularity_catalog_coverage_10": float(coverage.loc[coverage.model_name.eq("popularity") & coverage.k.eq(10), "catalog_coverage"].iloc[0]),
        "cold_target_share": metric("als", "full", "cold_target_share"),
        "als_training_seconds": bundle.metadata["fit_seconds"]["als"],
        "popularity_training_seconds": bundle.metadata["fit_seconds"]["popularity"],
        "reused_saved_model": reused, "train_fingerprint": inputs.train_fingerprint,
        "model_fingerprint": model_fingerprint(bundle), "dataset_fingerprint": evaluation_fingerprint(inputs)}


def add_baseline_deltas(frame):
    frame = frame.copy()
    base = frame.loc[frame.model_version.eq("1k")].iloc[0]
    for name in ("recall", "ndcg"):
        key = f"als_full_{name}_10"
        frame[f"{name}_delta_absolute"] = frame[key] - base[key]
        frame[f"{name}_delta_relative"] = frame[f"{name}_delta_absolute"] / base[key] if base[key] else None
    return frame


def run_comparison(root, *, train_new=False, validation_user_id=None, fitter=fit_models):
    root = Path(root).resolve()
    compatibility = check_comparison_compatibility(root)
    registry = load_version_registry(root)
    base_config = json.loads((root / "configs/recommendation_models.json").read_text(encoding="utf-8"))
    output = root / registry["comparison_dir"]
    if not output.resolve().is_relative_to(root / "artifacts/evaluation"):
        raise DataValidationError("Unsafe comparison output directory")
    rows, examples, identities, fingerprints = [], [], {}, {}
    baseline_items = None
    for version in ("1k", "5k", "10k"):
        entry = registry["versions"][version]
        inputs = load_training_inputs(root / entry["dataset_dir"])  # selective; no test access
        if inputs.dataset_config["positive_threshold"] != 5:
            raise DataValidationError("All models require unchanged threshold=5")
        identity = fixed_validation_identity(inputs)
        if identities and identity != next(iter(identities.values())):
            raise DataValidationError("Fixed validation changed between sample sizes")
        identities[version] = identity
        config = deepcopy(base_config)
        config.update(dataset_dir=entry["dataset_dir"], model_dir=entry["model_dir"],
                      diagnostic_dir=f"artifacts/validation/are008b/{version}")
        path = root / entry["model_dir"]
        reused = (path / "metadata.json").exists()
        if not reused:
            if version == "1k" or not train_new:
                raise DataValidationError(f"Model {version} missing; explicit --train-new required for new versions")
            if path.exists():
                raise DataValidationError("Incomplete model directory exists; refusing to overwrite")
            bundle = fitter(inputs, config)
            bundle.metadata["model_version"] = version
            # Central publisher owns staging, validation and no-replace locking.
            save_models(bundle, path)
        bundle = load_models(path, inputs.items, inputs.train_fingerprint)
        saved_config = bundle.metadata["training_config"]
        if saved_config["als"] != base_config["als"] or saved_config["blas_threads"] != base_config["blas_threads"]:
            raise DataValidationError("Saved model ALS/BLAS configuration differs from common experiment settings")
        if bundle.metadata["dataset_config"] != inputs.dataset_config:
            raise DataValidationError("Saved model dataset configuration differs")
        if baseline_items is not None and not baseline_items <= set(inputs.items.dataset_anime_id):
            raise DataValidationError("Train vocabulary is not nested")
        baseline_items = set(inputs.items.dataset_anime_id)
        report = evaluate_models(bundle, inputs, ks=(10,), include_als_fallback=False)
        report.metadata.update(franchise_filtering=False, model_version=version,
                               fixed_validation_fingerprint=identity)
        save_report(report, output / version)
        rows.append(comparison_row(version, inputs, bundle, report, reused))
        uid = int(inputs.validation_users.dataset_user_id.iloc[0]) if validation_user_id is None else validation_user_id
        catalog = pd.read_parquet(root / base_config["catalog_path"])
        if file_sha256(root / base_config["catalog_path"]) != inputs.catalog_sha256:
            raise DataValidationError("Catalog fingerprint differs")
        view = validation_view(ModelWorkspace(bundle, inputs, catalog), user_id=uid, top_k=10)
        selected = view["recommendations"].loc[lambda f: f.model_name.eq("als")].copy()
        selected["model_version"] = version
        examples.append(selected)
        fingerprints[version] = {"train": inputs.train_fingerprint, "model": model_fingerprint(bundle)}
    comparison = add_baseline_deltas(pd.DataFrame(rows))
    comparison_examples = pd.concat(examples, ignore_index=True)
    atomic_parquet(comparison, output / "comparison.parquet")
    atomic_parquet(comparison_examples, output / "validation_example.parquet")
    # Full NDCG@10 is the declared primary metric; Full Recall@10 breaks ties.
    best = comparison.sort_values(["als_full_ndcg_10", "als_full_recall_10"], ascending=False).iloc[0].model_version
    atomic_json(output / "comparison.json", {"split": "validation", "test_used": False,
        "franchise_filtering": False, "primary_metric": "ALS Full NDCG@10; Full Recall@10 tie-break",
        "recommended_model_version": best, "configured_default_version": registry["default_version"],
        "validation_user_id": uid, "fixed_validation_fingerprints": identities,
        "compatibility": compatibility,
        "model_fingerprints": fingerprints, "comparison": comparison.to_dict(orient="records"),
        "sha256": {name: file_sha256(output / name) for name in ("comparison.parquet", "validation_example.parquet")}})
    return comparison, comparison_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-new", action="store_true")
    parser.add_argument("--validation-user-id", type=int)
    args = parser.parse_args()
    frame, _ = run_comparison(Path.cwd(), train_new=args.train_new, validation_user_id=args.validation_user_id)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
