"""ARE-004 CLI: acquire -> validate -> clean -> local Parquet and quality report."""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sys

import requests

from anime_recommender.ingestion.training_data import acquire, atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError, clean_interactions, prepare_catalog


def validate_config(config, root):
    if not isinstance(config, dict):
        raise DataValidationError("Config must be a JSON object")
    required = {"source", "sample_users", "seed", "chunk_rows", "cache_dir", "output_dir", "cleaning"}
    if not required <= config.keys():
        raise DataValidationError(f"Missing config keys: {sorted(required - config.keys())}")
    for name in ["sample_users", "chunk_rows"]:
        if type(config[name]) is not int or config[name] < 1:
            raise DataValidationError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int:
        raise DataValidationError("seed must be an integer")
    for name in ["cache_dir", "output_dir"]:
        if not isinstance(config[name], str):
            raise DataValidationError(f"{name} must be a relative path string")
        path = Path(config[name])
        if path.is_absolute() or not (root / path).resolve().is_relative_to((root / "data").resolve()):
            raise DataValidationError(f"{name} must be a relative path inside data/")
    if (root / config["cache_dir"]).resolve() == (root / config["output_dir"]).resolve():
        raise DataValidationError("cache_dir and output_dir must differ")
    source = config["source"]
    if not isinstance(source, dict):
        raise DataValidationError("source must be an object")
    revision = source.get("revision")
    if (source.get("repository") != "mramazan/User-Animelist-Dataset"
            or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise DataValidationError("Expected official HF dataset with immutable commit revision")
    for kind in ["catalog", "ratings"]:
        if type(source.get(kind + "_bytes")) is not int or source[kind + "_bytes"] < 1:
            raise DataValidationError(f"Invalid source {kind} byte length")
        checksum = source.get(kind + "_sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise DataValidationError(f"Invalid source {kind} checksum")
    cleaning = config["cleaning"]
    if not isinstance(cleaning, dict):
        raise DataValidationError("cleaning must be an object")
    if cleaning.get("conflict_policy") != "drop_pair":
        raise DataValidationError("Only conflict_policy=drop_pair is supported without timestamps")
    bounds = [cleaning.get("rating_min"), cleaning.get("rating_max")]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
        raise DataValidationError("Invalid numeric rating bounds")


def _run_pipeline(config, root, allow_full_scan=False):
    raw, raw_catalog, provenance = acquire(config, root, allow_full_scan)
    catalog, mapping, quarantine, catalog_report = prepare_catalog(raw_catalog)
    clean, interaction_report = clean_interactions(raw, mapping,
        config["cleaning"]["rating_min"], config["cleaning"]["rating_max"])
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    tables = {"clean_interactions.parquet": clean, "anime_catalog.parquet": catalog,
              "anime_id_mapping.parquet": mapping, "catalog_quarantine.parquet": quarantine}
    # The quality report is the completion marker and is written last.
    for name, frame in tables.items():
        atomic_parquet(frame, output / name)
    report = {"status": "success", "pipeline_version": 1, "config": config, "provenance": provenance,
              "environment": {"python": platform.python_version(), "pandas": version("pandas"),
                  "pyarrow": version("pyarrow"), "requests": version("requests")},
              "dataset_attribution": {"author": "MRamazan", "license": "CC-BY-4.0",
                  "url": "https://huggingface.co/datasets/mramazan/User-Animelist-Dataset",
                  "changes": "complete-user hash sampling, conservative cleaning, MAL ID mapping"},
              "interactions": interaction_report, "catalog": catalog_report,
              "output_sha256": {name: file_sha256(output / name) for name in tables}}
    atomic_json(output / "data_quality_report.json", report)
    return report


def run_pipeline(config, root, allow_full_scan=False):
    root = Path(root).resolve()
    # Never write a failure report to an invalid/untrusted configured path.
    validate_config(config, root)
    try:
        return _run_pipeline(config, root, allow_full_scan)
    except (DataValidationError, OSError, ValueError, requests.RequestException) as error:
        output = root / config["output_dir"]
        message = str(error) if isinstance(error, DataValidationError) else type(error).__name__
        try:
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / "data_quality_report.json", {
                "status": "failed", "pipeline_version": 1, "config": config,
                "error": message, "note": "Incomplete run; existing output tables are not certified by this report",
            })
        except OSError:
            # Preserve the original actionable error if storage itself failed.
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_data.json")
    parser.add_argument("--allow-full-scan", action="store_true",
                        help="Explicitly permit one full source scan when no valid sample cache exists")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    report = run_pipeline(config, root, args.allow_full_scan)
    print(json.dumps({"interactions": report["interactions"], "catalog": report["catalog"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        print("Pipeline failed: source network/HTTP error; no automatic retry", file=sys.stderr)
        raise SystemExit(1) from None
    except (DataValidationError, OSError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
