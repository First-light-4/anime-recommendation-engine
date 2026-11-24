"""Verified model format v1 and immutable, cooperating-writer-safe publication."""
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits
from anime_recommender.artifact_files import atomic_json, atomic_parquet, file_sha256, publish_directory
from anime_recommender.contracts import DataValidationError, validate_factors, validate_item_mapping
from anime_recommender.inference.bundle import ModelBundle
from anime_recommender.inference.popularity import PopularityRecommender

BUNDLE_FILES = frozenset({"als.npz", "item_mapping.parquet", "train_user_mapping.parquet", "popularity.parquet", "training_config.json", "metadata.json"})


def read_model_metadata(directory):
    directory = Path(directory)
    missing = sorted(name for name in BUNDLE_FILES if not (directory / name).is_file())
    if missing:
        raise FileNotFoundError(f"Model bundle missing/incomplete at {directory}: {', '.join(missing)}")
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise DataValidationError("Invalid model metadata JSON") from error
    if not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int or metadata["format_version"] != 1:
        raise DataValidationError("Unsupported model artifact format_version; require v1")
    for key in ("training_config", "dataset_config", "files_sha256"):
        if not isinstance(metadata.get(key), dict):
            raise DataValidationError(f"Missing/invalid model metadata: {key}")
    if not isinstance(metadata.get("train_fingerprint"), str) or not metadata["train_fingerprint"]:
        raise DataValidationError("Missing model train fingerprint")
    return metadata


_publish_directory = publish_directory  # compatibility test seam


def save_models(bundle, directory):
    """Publish a complete new v1 bundle, never overwrite or implicitly reuse.

    O_EXCL sibling lock coordinates publishers, then native no-replace rename
    protects even against a destination created by an uncooperative writer.
    Stale locks after process termination require explicit operator inspection.
    """
    destination = Path(directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Model bundle already exists; immutable publication: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ("." + destination.name + ".publish.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError(f"Model publication locked; inspect active/stale publisher: {destination}") from None
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Model bundle already exists: {destination}")
        with TemporaryDirectory(prefix="." + destination.name + "-staging-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            metadata = copy.deepcopy(bundle.metadata)
            if metadata.get("format_version") != 1:
                raise DataValidationError("Only model artifact format v1 can be published")
            bundle.als.save(staging / "als.npz")
            atomic_parquet(bundle.items, staging / "item_mapping.parquet")
            atomic_parquet(bundle.train_users, staging / "train_user_mapping.parquet")
            atomic_parquet(bundle.popularity, staging / "popularity.parquet")
            atomic_json(staging / "training_config.json", metadata["training_config"])
            metadata["mapping_fingerprint"] = mapping_fingerprint(bundle.items)
            metadata["files_sha256"] = {name: file_sha256(staging / name) for name in (
                "als.npz", "item_mapping.parquet", "train_user_mapping.parquet", "popularity.parquet", "training_config.json")}
            atomic_json(staging / "metadata.json", metadata)  # completion marker last
            load_models(staging, bundle.items, metadata["train_fingerprint"])
            _publish_directory(staging, destination)
        # Preserve legacy successful-save metadata enrichment, never on failure.
        bundle.metadata.update(metadata)
    finally:
        lock.unlink()

def mapping_fingerprint(mapping):
    values = validate_item_mapping(mapping).to_numpy(dtype="int64").tolist()
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def load_models(directory, expected_mapping=None, expected_train_fingerprint=None) -> ModelBundle:
    """Read and verify v1; report malformed payloads without backend trace details."""
    try:
        return _load_models(directory, expected_mapping, expected_train_fingerprint)
    except (DataValidationError, FileNotFoundError):
        raise
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        raise DataValidationError(f"Invalid model bundle at {directory}: {type(error).__name__}") from error


def _load_models(directory, expected_mapping=None, expected_train_fingerprint=None):
    directory = Path(directory)
    metadata = read_model_metadata(directory)
    expected_files = {"als.npz", "item_mapping.parquet", "train_user_mapping.parquet", "popularity.parquet", "training_config.json"}
    if set(metadata.get("files_sha256", {})) != expected_files:
        raise DataValidationError("Unexpected model bundle file list")
    for name, sha in metadata["files_sha256"].items():
        if file_sha256(directory / name) != sha:
            raise DataValidationError(f"Model checksum mismatch: {name}")
    saved_config = json.loads((directory / "training_config.json").read_text(encoding="utf-8"))
    if saved_config != metadata.get("training_config"):
        raise DataValidationError("Saved configuration and model metadata differ")
    mapping = validate_item_mapping(pd.read_parquet(directory / "item_mapping.parquet"))
    if mapping_fingerprint(mapping) != metadata.get("mapping_fingerprint"):
        raise DataValidationError("Model mapping fingerprint mismatch")
    if expected_mapping is not None and mapping_fingerprint(expected_mapping) != metadata["mapping_fingerprint"]:
        raise DataValidationError("Supplied item mapping is from a different model/dataset")
    if expected_train_fingerprint is not None and metadata.get("train_fingerprint") != expected_train_fingerprint:
        raise DataValidationError("Model belongs to a different training dataset")
    with threadpool_limits(limits=1, user_api="blas"):
        # implicit 0.7.3 load uses np.load(..., allow_pickle=False).
        als = AlternatingLeastSquares.load(directory / "als.npz")
    users = pd.read_parquet(directory / "train_user_mapping.parquet")
    validate_factors(als, len(users), len(mapping))
    config = metadata["training_config"]["als"]
    for key in config:
        actual = getattr(als, key)
        if str(actual) != str(config[key]):
            raise DataValidationError(f"Loaded ALS parameter mismatch: {key}")
    popularity = pd.read_parquet(directory / "popularity.parquet")
    PopularityRecommender(popularity, mapping)  # validate saved alignment
    return ModelBundle(als, popularity, mapping, users, metadata)
