"""Single shared maximum-size acquisition, followed by local nested preparation."""

import json
from pathlib import Path

import pandas as pd

from anime_recommender.ingestion.config import validate_config
from anime_recommender.ingestion.training_data import acquire, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError, clean_interactions, prepare_catalog
from anime_recommender.preprocessing.dataset import load_dataset
from anime_recommender.preprocessing.scaling import (PROTOCOL, _same, fingerprint, nested_users,
    prepare_scaling, register_baseline, save_scaling_dataset, scaling_summary)


def run_scaling(root, config_path="configs/scaling.json", *, execute=False, allow_full_scan=False, acquirer=acquire):
    root = Path(root).resolve()
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    for key in ("source_config", "baseline_dir", "output_dir", "cache_dir"):
        path = (root / config[key]).resolve()
        if not path.is_relative_to(root) or (key != "source_config" and not path.is_relative_to(root / "data")):
            raise DataValidationError(f"Unsafe scaling path: {key}")
    baseline_dir = root / config["baseline_dir"]
    baseline = load_dataset(baseline_dir)
    source_config = json.loads((root / config["source_config"]).read_text(encoding="utf-8"))
    sizes = config["sizes"]
    baseline_ids = set(baseline.tables["user_assignments"].dataset_user_id)
    if (not sizes or any(type(size) is not int or size <= len(baseline_ids) for size in sizes)
            or sizes != sorted(set(sizes))):
        raise DataValidationError("Expansion sizes must be unique, increasing and larger than baseline")
    output = root / config["output_dir"]
    cache = root / config["cache_dir"]
    if output == baseline_dir or output in baseline_dir.parents or cache == root / source_config["cache_dir"]:
        raise DataValidationError("Scaling must not overwrite baseline/cache")
    # Verify provenance before accessing the network or producing output.
    baseline_source = json.loads((root / source_config["output_dir"] / "data_quality_report.json").read_text(encoding="utf-8"))
    if baseline_source.get("status") != "success" or baseline_source["config"] != source_config:
        raise DataValidationError("Source configuration differs from certified baseline")
    if baseline.report.get("source_sha256") != baseline_source["output_sha256"]:
        raise DataValidationError("Baseline source fingerprint mismatch")
    for filename, expected in baseline_source["output_sha256"].items():
        if file_sha256(root / source_config["output_dir"] / filename) != expected:
            raise DataValidationError("Baseline cleaned data fingerprint mismatch")
    baseline_raw, _, old_manifest = acquirer(source_config, root, False)
    if set(baseline_raw.userID.astype("int64")) != baseline_ids:
        raise DataValidationError("Raw sampling and baseline assignments differ")
    maximum_config = dict(source_config, sample_users=max(sizes), cache_dir=config["cache_dir"])
    validate_config(maximum_config, root)
    plan = {"mode": "plan", "protocol": PROTOCOL, "sizes": sizes,
        "source_bytes": source_config["source"]["ratings_bytes"],
        "full_passes_needed": 0 if (cache / "manifest.json").exists() else 1,
        "resume": "Completed cache/experiments reused; interrupted source scan restarts from byte zero",
        "network_started": False}
    if not execute:
        return plan
    # acquire refuses any uncached source scan unless explicitly authorized.
    raw, catalog_raw, provenance = acquirer(maximum_config, root, allow_full_scan)
    if provenance["parameters"]["source"] != old_manifest["parameters"]["source"]:
        raise DataValidationError("Expanded source fingerprint differs")
    if (provenance.get("source_ratings_sha256") != source_config["source"]["ratings_sha256"]
            or provenance.get("source_ratings_bytes") != source_config["source"]["ratings_bytes"]):
        raise DataValidationError("Expanded histories lack a verified complete source scan")
    catalog, mapping, _, _ = prepare_catalog(catalog_raw)
    del catalog
    baseline_clean = pd.read_parquet(root / source_config["output_dir"] / "clean_interactions.parquet")
    baseline_name = f"baseline_{len(baseline_ids)}"
    outputs = {baseline_name: register_baseline(baseline_dir, output / baseline_name)}
    previous = None
    previous_selected = baseline_ids
    for size in sizes:
        selected = nested_users(raw.userID.astype("int64").unique(), baseline_ids, size, source_config["seed"])
        if not previous_selected <= set(selected):
            raise DataValidationError("Selected populations are not nested")
        subset = raw.loc[raw.userID.astype("int64").isin(selected)]
        clean, quality = clean_interactions(subset, mapping,
            source_config["cleaning"]["rating_min"], source_config["cleaning"]["rating_max"])
        _same(clean.loc[clean.dataset_user_id.isin(baseline_ids)].reset_index(drop=True), baseline_clean, "all baseline histories")
        prepared = prepare_scaling(clean, mapping, baseline, selected, baseline.report["config"], previous=previous)
        manifest = {"protocol": PROTOCOL, "strategy": "baseline-preserving-sha256-bottom-k", "seed": source_config["seed"],
            "sample_size": size, "selected_user_ids": selected,
            "source": source_config["source"], "source_cache_sha256": provenance["cache_sha256"],
            "baseline_report_sha256": file_sha256(baseline_dir / "dataset_report.json"),
            "config_fingerprint": fingerprint(baseline.report["config"])}
        prepared.report["cleaning"] = quality
        prepared.report["source_sha256"] = baseline.report["source_sha256"]
        name = f"sample_{size}"
        outputs[name] = save_scaling_dataset(output / name, prepared, manifest, baseline_directory=baseline_dir)
        previous, previous_selected = prepared, set(selected)
    return {"mode": "prepared", "statistics": scaling_summary(outputs).to_dict(orient="records")}
