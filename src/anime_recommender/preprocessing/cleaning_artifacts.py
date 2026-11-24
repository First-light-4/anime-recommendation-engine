"""Local raw-to-clean tables and ARE-004 quality manifest; no acquisition."""
from importlib.metadata import version
import platform
from anime_recommender.artifact_files import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import clean_interactions, prepare_catalog

def clean_raw_dataset(config, root, raw, raw_catalog, provenance):
    return write_clean_raw(config, root / config["output_dir"], raw, raw_catalog, provenance)


def write_clean_raw(config, output, raw, raw_catalog, provenance):
    catalog, mapping, quarantine, catalog_report = prepare_catalog(raw_catalog)
    clean, interaction_report = clean_interactions(raw, mapping,
        config["cleaning"]["rating_min"], config["cleaning"]["rating_max"])
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
