"""Numeric Parquet tables and strict JSON manifests; local validation data only."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from anime_recommender.artifact_files import atomic_json, atomic_parquet, file_sha256


TABLES = {"metrics", "cohorts", "user_metrics", "users", "history_distribution", "coverage",
          "recommendations", "statuses", "examples", "example_users"}


@dataclass
class EvaluationReport:
    metadata: dict
    tables: dict[str, pd.DataFrame]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_report(report, directory):
    if set(report.tables) != TABLES:
        raise ValueError("Unexpected evaluation table set")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in report.tables.items():
        atomic_parquet(frame, directory / (name + ".parquet"))
    manifest = {"metadata": report.metadata,
        "metrics": report.tables["metrics"].to_dict("records"),
        "cohorts": report.tables["cohorts"].to_dict("records"),
        "coverage": report.tables["coverage"].to_dict("records"),
        "history_distribution": report.tables["history_distribution"].to_dict("records"),
        "files_sha256": {name + ".parquet": file_sha256(directory / (name + ".parquet")) for name in sorted(TABLES)}}
    manifest = json_safe(manifest)
    json.dumps(manifest, allow_nan=False)  # Reject any non-JSON value before writing.
    atomic_json(directory / "report.json", manifest)


def load_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    hashes = manifest["files_sha256"]
    if set(hashes) != {name + ".parquet" for name in TABLES}:
        raise ValueError("Unexpected report file set")
    for name, digest in hashes.items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Evaluation report checksum mismatch")
    tables = {name: pd.read_parquet(directory / (name + ".parquet")) for name in TABLES}
    for name in ("metrics", "cohorts", "coverage", "history_distribution"):
        if json_safe(tables[name].to_dict("records")) != manifest[name]:
            raise ValueError("JSON and Parquet report disagree")
    return EvaluationReport(manifest["metadata"], tables)
