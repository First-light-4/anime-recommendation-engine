"""Explicit saved-model selection. No dataset rebuilding, training or discovery."""

import json
from pathlib import Path

from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.recommendation_exploration import load_profile_workspace


def load_version_registry(root):
    root = Path(root).resolve()
    config = json.loads((root / "configs/model_versions.json").read_text(encoding="utf-8"))
    if set(config["versions"]) != {"1k", "5k", "10k"} or config["default_version"] not in config["versions"]:
        raise DataValidationError("Invalid model version registry")
    for entry in config["versions"].values():
        for key, parent in (("model_dir", "artifacts/models"), ("dataset_dir", "data")):
            path = (root / entry[key]).resolve()
            if not path.is_relative_to((root / parent).resolve()):
                raise DataValidationError("Version path escapes local artifact directories")
    return config


def load_version_workspace(root, model_version=None):
    registry = load_version_registry(root)
    if model_version is None:
        model_version = registry["default_version"]
    if model_version not in registry["versions"]:
        raise DataValidationError("model_version must be one of: 1k, 5k, 10k")
    path = Path(root) / registry["versions"][model_version]["model_dir"]
    if not (path / "metadata.json").is_file():
        raise DataValidationError(f"Saved model {model_version} is missing; train it separately. No automatic fallback.")
    workspace = load_profile_workspace(root, model_dir=path)
    if model_version != "1k" and workspace.bundle.metadata.get("model_version") != model_version:
        raise DataValidationError("Saved artifact version differs from selected model_version")
    return workspace


def version_franchise_coverage(workspace, model_version):
    items = workspace.bundle.items
    mapping = workspace.franchise_mapping
    # Membership is defined by structured local metadata only, never inferred from titles.
    from anime_recommender.ingestion.franchise_artifacts import candidate_metadata_coverage
    return {"model_version": model_version, "train_items": len(items),
            **(candidate_metadata_coverage(mapping, items) if mapping is not None else
               {"covered_occurrences": 0, "covered_candidate_share": 0.0, "known_candidate_groups": 0}),
            "metadata_source": (workspace.franchise_config or {}).get("metadata_source"),
            "warning": (workspace.franchise_config or {}).get("metadata_warning")}
