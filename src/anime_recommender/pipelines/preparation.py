"""Independent local raw-to-clean and clean-to-splits operations."""
from pathlib import Path
from tempfile import TemporaryDirectory

from anime_recommender.artifact_files import file_sha256, publish_directory
from anime_recommender.contracts import DataValidationError
from anime_recommender.ingestion.config import validate_config as validate_source_config
from anime_recommender.ingestion.processed_data import load_processed_data
from anime_recommender.ingestion.training_data import load_cached_raw
from anime_recommender.preprocessing.cleaning_artifacts import write_clean_raw
from anime_recommender.preprocessing.dataset import (
    load_dataset, prepare_dataset, save_prepared_dataset, validate_config,
)


def run_cleaning(config, root):
    root = Path(root).resolve()
    validate_source_config(config, root)
    raw, catalog, provenance = load_cached_raw(config, root)
    output = root / config["output_dir"]
    if output.exists():
        existing = load_processed_data(output)
        report = existing.quality_report
        if report["config"] != config or report["provenance"] != provenance:
            raise DataValidationError("Processed dataset identity differs; select a new output_dir")
        for name, expected in report["output_sha256"].items():
            if file_sha256(output / name) != expected:
                raise DataValidationError(f"Processed dataset checksum mismatch: {name}")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cleaning-", dir=output.parent) as temporary:
        # Preserve configured identity in metadata; only storage path is staged.
        report = write_clean_raw(config, Path(temporary), raw, catalog, provenance)
        load_processed_data(temporary)
        publish_directory(temporary, output)
    return report


def run_preparation(config, root):
    root = Path(root).resolve()
    validate_config(config, root)
    data = load_processed_data(root / config["input_dir"])
    output = root / config["output_dir"]
    if output.exists():
        return load_dataset(output, expected_config=config,
                            expected_source_sha256=data.quality_report["output_sha256"])
    prepared = prepare_dataset(data.interactions, data.mapping, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".preparation-", dir=output.parent) as temporary:
        staging = Path(temporary)
        save_prepared_dataset(prepared, staging, config, data.quality_report["output_sha256"])
        load_dataset(staging, expected_config=config,
                     expected_source_sha256=data.quality_report["output_sha256"])
        publish_directory(staging, output)
    return prepared
