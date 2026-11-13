"""Selective ARE-005.1 reader: train and validation only, never opens test artifacts."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz

from anime_recommender.ingestion.training_data import file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError


@dataclass
class TrainingInputs:
    train: pd.DataFrame
    train_users: pd.DataFrame
    items: pd.DataFrame
    matrix: object
    validation_context: pd.DataFrame
    validation_target: pd.DataFrame
    observed_context: pd.DataFrame
    validation_users: pd.DataFrame
    dataset_config: dict
    train_fingerprint: str
    train_hashes: dict
    catalog_sha256: str


def load_training_inputs(directory) -> TrainingInputs:
    directory = Path(directory)
    report = json.loads((directory / "dataset_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "success" or report.get("dataset_version") != 2:
        raise DataValidationError("Require successful ARE-005.1 v2 artifacts")
    names = ["train_interactions.parquet", "train_user_mapping.parquet", "item_mapping.parquet",
             "train_matrix.npz", "validation_context.parquet", "validation_target.parquet",
             "validation_observed_context.parquet", "validation_user_mapping.parquet"]
    for name in names:
        if file_sha256(directory / name) != report.get("artifact_sha256", {}).get(name):
            raise DataValidationError(f"Dataset checksum mismatch: {name}")
    train_hashes = {name: report["artifact_sha256"][name] for name in names[:4]}
    payload = {"train_hashes": train_hashes, "dataset_config": report["config"]}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    frames = {name: pd.read_parquet(directory / name) for name in names if name.endswith(".parquet")}
    train = frames["train_interactions.parquet"]
    train_users = frames["train_user_mapping.parquet"]
    items = frames["item_mapping.parquet"]
    matrix = load_npz(directory / "train_matrix.npz")
    validation_users = frames["validation_user_mapping.parquet"]
    if set(train_users.dataset_user_id) & set(validation_users.dataset_user_id):
        raise DataValidationError("Train-validation user leakage")
    if matrix.shape != (len(train_users), len(items)) or matrix.nnz != len(train):
        raise DataValidationError("Train CSR and mappings do not agree")
    return TrainingInputs(train, train_users, items, matrix,
        frames["validation_context.parquet"], frames["validation_target.parquet"],
        frames["validation_observed_context.parquet"], validation_users,
        report["config"], fingerprint, train_hashes,
        report["source_sha256"]["anime_catalog.parquet"])
