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


from anime_recommender.ingestion.config import validate_config
from anime_recommender.preprocessing.cleaning_artifacts import clean_raw_dataset


def _run_pipeline(config, root, allow_full_scan=False):
    raw, catalog, provenance = acquire(config, root, allow_full_scan)
    return clean_raw_dataset(config, root, raw, catalog, provenance)


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
