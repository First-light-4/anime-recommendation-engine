"""Validation-only reader. Train identity/vocabulary/user IDs come from saved bundle."""
from dataclasses import dataclass
import pandas as pd

from anime_recommender.artifact_io import mapping_fingerprint
from anime_recommender.contracts import DataValidationError
from anime_recommender.dataset_artifacts import (
    VALIDATION_FILES, dataset_manifest, read_tables, training_identity,
)


@dataclass
class ValidationView:
    validation_context: pd.DataFrame
    validation_target: pd.DataFrame
    observed_context: pd.DataFrame
    validation_users: pd.DataFrame
    items: pd.DataFrame
    train_users: pd.DataFrame
    dataset_config: dict
    train_fingerprint: str
    catalog_sha256: str


def load_validation_view(directory, bundle) -> ValidationView:
    directory, report = dataset_manifest(directory)
    fingerprint, _ = training_identity(report)
    if fingerprint != bundle.metadata["train_fingerprint"]:
        raise DataValidationError("Evaluation/model training fingerprint mismatch")
    if mapping_fingerprint(bundle.items) != bundle.metadata["mapping_fingerprint"]:
        raise DataValidationError("Evaluation/model mapping fingerprint mismatch")
    tables = read_tables(directory, report, VALIDATION_FILES)
    context, target, observed, users = (tables[name] for name in VALIDATION_FILES)
    if set(users.dataset_user_id) & set(bundle.train_users.dataset_user_id):
        raise DataValidationError("Train-validation user leakage")
    return ValidationView(context, target, observed, users, bundle.items.copy(),
        bundle.train_users.copy(), report["config"], fingerprint,
        report["source_sha256"]["anime_catalog.parquet"])
