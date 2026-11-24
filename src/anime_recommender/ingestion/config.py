"""Source acquisition configuration checks; no stage execution."""
import math
import re
from pathlib import Path
from anime_recommender.contracts import DataValidationError

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
