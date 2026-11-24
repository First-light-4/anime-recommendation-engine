"""Read-only scaling and comparison inspection; no training runner imports."""
import json
from pathlib import Path
import pandas as pd
from anime_recommender.artifact_files import file_sha256
from anime_recommender.artifact_io import load_models
from anime_recommender.contracts import DataValidationError
from anime_recommender.registry.models import load_version_registry
from anime_recommender.research_inputs import load_training_inputs

def fixed_validation_identity(inputs):
    """Only the invariant source fields, excluding changing warm/cold diagnostics."""
    import hashlib
    digest = hashlib.sha256()
    for frame in (inputs.observed_context, inputs.validation_context, inputs.validation_target):
        frame = frame.drop(columns="is_cold_item", errors="ignore").sort_values(["dataset_user_id", "dataset_anime_id"])
        digest.update(frame.to_json(orient="split", index=False, double_precision=15).encode())
    digest.update(json.dumps(sorted(map(int, inputs.validation_users.dataset_user_id))).encode())
    return digest.hexdigest()


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
