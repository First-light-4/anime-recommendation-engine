"""Read and verify existing ARE-004 artifacts. Never downloads or rewrites data."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from anime_recommender.ingestion.training_data import file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError, require_columns


@dataclass
class ProcessedData:
    interactions: pd.DataFrame
    catalog: pd.DataFrame
    mapping: pd.DataFrame
    quality_report: dict


def load_processed_data(directory: str | Path) -> ProcessedData:
    directory = Path(directory)
    report = json.loads((directory / "data_quality_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise DataValidationError("ARE-004 report does not certify a successful run")
    names = ["clean_interactions.parquet", "anime_catalog.parquet", "anime_id_mapping.parquet"]
    for name in names:
        path = directory / name
        if not path.is_file() or file_sha256(path) != report.get("output_sha256", {}).get(name):
            raise DataValidationError(f"Missing or checksum-mismatched processed artifact: {name}")
    interactions, catalog, mapping = [pd.read_parquet(directory / name) for name in names]
    require_columns(interactions, ["dataset_user_id", "dataset_anime_id", "rating", "myanimelist_id"], "interactions")
    require_columns(catalog, ["dataset_anime_id", "myanimelist_id", "title"], "catalog")
    require_columns(mapping, ["dataset_anime_id", "myanimelist_id"], "mapping")
    return ProcessedData(interactions, catalog, mapping, report)
