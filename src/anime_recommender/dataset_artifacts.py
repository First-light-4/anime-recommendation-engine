"""Selective v2 artifact verification. Never opens unrequested split files."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from anime_recommender.artifact_files import file_sha256
from anime_recommender.contracts import DataValidationError

TRAIN_FILES = ("train_interactions.parquet", "train_user_mapping.parquet",
               "item_mapping.parquet", "train_matrix.npz")
VALIDATION_FILES = ("validation_context.parquet", "validation_target.parquet",
                    "validation_observed_context.parquet", "validation_user_mapping.parquet")


def dataset_manifest(directory):
    directory = Path(directory).resolve()
    visited = set()
    while True:
        if directory in visited:
            raise DataValidationError("Cyclic baseline reference")
        visited.add(directory)
        report = json.loads((directory / "dataset_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "success" or report.get("dataset_version") != 2:
            raise DataValidationError("Require successful ARE-005.1 v2 artifacts")
        if "baseline_reference" not in report:
            return directory, report
        reference = Path(report["baseline_reference"]).resolve()
        if file_sha256(reference / "dataset_report.json") != report.get("baseline_report_sha256"):
            raise DataValidationError("Baseline reference fingerprint mismatch")
        directory = reference


def training_identity(report):
    hashes = {name: report["artifact_sha256"][name] for name in TRAIN_FILES}
    payload = {"train_hashes": hashes, "dataset_config": report["config"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), hashes


def verified_path(directory, report, name):
    reference = report.get("shared_artifacts", {}).get(name)
    path = Path(reference["path"]) if reference else directory / name
    expected = reference["sha256"] if reference else report.get("artifact_sha256", {}).get(name)
    if expected is None or file_sha256(path) != expected:
        raise DataValidationError(f"Dataset checksum mismatch: {name}")
    return path


def read_tables(directory, report, names):
    return {name: pd.read_parquet(verified_path(directory, report, name)) for name in names}
