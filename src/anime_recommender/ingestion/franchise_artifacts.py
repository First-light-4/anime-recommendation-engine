"""Local-only verified metadata artifact reader; no HTTP or training imports."""

import json
from pathlib import Path

from anime_recommender.ingestion.training_data import file_sha256


def load_prepared_mapping(directory):
    directory = Path(directory).resolve()
    latest = json.loads((directory / "latest.json").read_text())
    version = (directory / latest["version"]).resolve()
    if latest.get("schema_version") != 1 or not version.is_relative_to(directory / "versions"):
        raise ValueError("Invalid prepared metadata pointer")
    if file_sha256(version / "manifest.json") != latest["manifest_sha256"]:
        raise ValueError("Prepared metadata manifest checksum mismatch")
    manifest = json.loads((version / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared metadata schema")
    if set(manifest["files_sha256"]) != {"mapping.json", "parts.parquet", "coverage.json", "source_records.json"}:
        raise ValueError("Unexpected prepared metadata files")
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(version / name) != expected:
            raise ValueError("Prepared metadata checksum mismatch")
    return json.loads((version / "mapping.json").read_text()), json.loads((version / "coverage.json").read_text())


def candidate_metadata_coverage(mapping, recommendations):
    """Coverage over supplied candidate occurrences, not a ranking quality metric."""
    count = len(recommendations)
    known = recommendations.myanimelist_id.isin(mapping.franchise)
    return {"candidate_occurrences": count, "covered_occurrences": int(known.sum()),
        "covered_candidate_share": float(known.mean()) if count else None,
        "unique_candidate_items": int(recommendations.myanimelist_id.nunique()),
        "known_candidate_groups": len({mapping.franchise[m] for m in recommendations.loc[known, "myanimelist_id"]})}
