"""Explicit saved-model selection. No dataset rebuilding, training or discovery."""

from pathlib import Path

from anime_recommender.registry.models import ModelRegistry, load_version_registry
from anime_recommender.registry.workspace import load_profile_workspace


def load_version_workspace(root, model_version=None):
    registry = ModelRegistry(root)
    version, entry = registry.resolve(model_version)
    bundle = registry.load(version)
    return load_profile_workspace(root, model_dir=Path(root) / entry["model_dir"],
                                  model_loader=lambda _: bundle)


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
