"""ARE-008B: identical ALS settings, fixed validation, no franchise filtering/test."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from anime_recommender.evaluation.ranking_evaluator import evaluate_models, evaluation_fingerprint, model_fingerprint
from anime_recommender.evaluation.reports import save_report
from anime_recommender.inference.model_versions import load_version_registry
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.recommendation_exploration import ModelWorkspace, validation_view
from anime_recommender.training.artifacts import load_models, save_models
from anime_recommender.training.inputs import load_training_inputs
from anime_recommender.training.pipeline import fit_models


def fixed_validation_identity(inputs):
    """Only the invariant source fields, excluding changing warm/cold diagnostics."""
    import hashlib
    digest = hashlib.sha256()
    for frame in (inputs.observed_context, inputs.validation_context, inputs.validation_target):
        frame = frame.drop(columns="is_cold_item", errors="ignore").sort_values(["dataset_user_id", "dataset_anime_id"])
        digest.update(frame.to_json(orient="split", index=False, double_precision=15).encode())
    digest.update(json.dumps(sorted(map(int, inputs.validation_users.dataset_user_id))).encode())
    return digest.hexdigest()


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


def check_comparison_compatibility(root):
    """Validate all three inputs and baseline reuse before fitting any new model."""
    root = Path(root)
    registry = load_version_registry(root)
    common = json.loads((root / "configs/recommendation_models.json").read_text(encoding="utf-8"))
    checks, identity, previous_users, previous_items, configuration = {}, None, set(), set(), None
    for version in ("1k", "5k", "10k"):
        entry = registry["versions"][version]
        inputs = load_training_inputs(root / entry["dataset_dir"])
        current = fixed_validation_identity(inputs)
        if identity is not None and current != identity:
            raise DataValidationError("Fixed validation differs across versions")
        identity = current
        if inputs.dataset_config["positive_threshold"] != 5 or any(inputs.dataset_config[k] != 5 for k in ("min_user_positives", "min_item_users")):
            raise DataValidationError("Expected threshold=5, train filtering=5/5")
        if configuration is not None and inputs.dataset_config != configuration:
            raise DataValidationError("Dataset preparation configurations differ")
        configuration = inputs.dataset_config
        users, items = set(inputs.train_users.dataset_user_id), set(inputs.items.dataset_anime_id)
        if not previous_users <= users or not previous_items <= items:
            raise DataValidationError("Train users/vocabularies are not nested")
        previous_users, previous_items = users, items
        if version == "1k":
            saved = load_models(root / entry["model_dir"], inputs.items, inputs.train_fingerprint)
            if saved.metadata["training_config"]["als"] != common["als"] or saved.metadata["training_config"]["blas_threads"] != common["blas_threads"]:
                raise DataValidationError("Baseline hyperparameters differ; refusing baseline retraining")
        checks[version] = {"shape": list(inputs.matrix.shape), "nnz": inputs.matrix.nnz,
            "train_fingerprint": inputs.train_fingerprint, "fixed_validation_fingerprint": current}
    return checks


def run_comparison(root, *, train_new=False, validation_user_id=None):
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
            bundle = fit_models(inputs, config)
            bundle.metadata["model_version"] = version
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".model-", dir=path.parent) as temporary:
                staging = Path(temporary)
                save_models(bundle, staging)
                load_models(staging, inputs.items, inputs.train_fingerprint)
                staging.replace(path)
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


def load_comparison(root):
    registry = load_version_registry(root)
    output = Path(root) / registry["comparison_dir"]
    metadata = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    for name, expected in metadata["sha256"].items():
        if file_sha256(output / name) != expected:
            raise DataValidationError("Comparison artifact checksum mismatch")
    return {"metadata": metadata, "comparison": pd.read_parquet(output / "comparison.parquet"),
            "validation_example": pd.read_parquet(output / "validation_example.parquet")}


def selective_scaling_overview(root):
    """Notebook overview of train/validation only; never invokes ARE-008A full loader."""
    root = Path(root)
    registry = load_version_registry(root)
    rows, activity = [], []
    for version in ("1k", "5k", "10k"):
        entry = registry["versions"][version]
        inputs = load_training_inputs(root / entry["dataset_dir"])
        context, target = inputs.validation_context, inputs.validation_target
        report = json.loads((root / entry["dataset_dir"] / "dataset_report.json").read_text(encoding="utf-8"))
        rows.append({"model_version": version, "selected_users": sum(c["allocated_users"] for c in report["cohorts"].values()),
            "clean_interactions": report["input"]["rows"], "positive_interactions": report["input"]["positive_rows"],
            "train_users": len(inputs.train_users), "train_items": len(inputs.items), "train_positives": len(inputs.train),
            "train_csr_shape": str(inputs.matrix.shape), "train_csr_nnz": inputs.matrix.nnz,
            "sparsity": 1 - inputs.matrix.nnz / (inputs.matrix.shape[0] * inputs.matrix.shape[1]),
            "validation_context_cold_share": float(context.is_cold_item.mean()),
            "validation_target_cold_share": float(target.is_cold_item.mean()),
            "validation_warm_context_nnz": int((~context.is_cold_item).sum()),
            "train_fingerprint": inputs.train_fingerprint})
        counts = inputs.train.groupby("dataset_user_id").size()
        activity.append({"model_version": version, **counts.describe().to_dict()})
    return {"summary": pd.DataFrame(rows), "train_activity": pd.DataFrame(activity)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-new", action="store_true")
    parser.add_argument("--validation-user-id", type=int)
    args = parser.parse_args()
    frame, _ = run_comparison(Path.cwd(), train_new=args.train_new, validation_user_id=args.validation_user_id)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
