"""Configuration-backed local registry. Adding versions requires no Python edits."""

from dataclasses import dataclass
import json
from pathlib import Path

from anime_recommender.artifact_io import BUNDLE_FILES, load_models, read_model_metadata
from anime_recommender.contracts import DataValidationError, ModelIdentity, ModelMetadata


@dataclass(frozen=True)
class ModelRecord:
    identity: ModelIdentity
    artifact_path: Path
    available: bool
    metadata: ModelMetadata | None


def load_version_registry(root, config_path="configs/model_versions.json"):
    root = Path(root).resolve()
    config_file = (root / config_path).resolve()
    if not config_file.is_relative_to(root):
        raise DataValidationError("Registry config must stay inside project root")
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise DataValidationError("Invalid registry JSON") from error
    if not isinstance(config, dict):
        raise DataValidationError("Registry config must be an object")
    if type(config.get("schema_version", 1)) is not int or config.get("schema_version", 1) != 1:
        raise DataValidationError("Unsupported registry schema_version")
    versions = config.get("versions")
    if not isinstance(versions, dict) or not versions or config.get("default_version") not in versions:
        raise DataValidationError("Invalid model version registry/default")
    ids = set()
    for version, entry in versions.items():
        if not isinstance(version, str) or not version or not isinstance(entry, dict):
            raise DataValidationError("Invalid registry version entry")
        model_id = entry.get("model_id", version)
        if not isinstance(model_id, str) or not model_id or model_id in ids:
            raise DataValidationError("Model IDs must be unique nonempty strings")
        ids.add(model_id)
        for field in ("train_fingerprint", "mapping_fingerprint", "artifact_version"):
            if field in entry and (not isinstance(entry[field], str) or not entry[field]):
                raise DataValidationError(f"Invalid registry {field}")
        if entry.get("model_type", "als") != "als":
            raise DataValidationError("Unsupported model type; v1 bundles contain ALS and Popularity")
        for key, parent in (("model_dir", "artifacts/models"), ("dataset_dir", "data")):
            if key == "dataset_dir" and key not in entry:
                continue
            if not isinstance(entry.get(key), str) or not (root / entry[key]).resolve().is_relative_to(root / parent):
                raise DataValidationError("Version path escapes local artifact directories")
    # Ambiguous lookup by alias vs another entry's model_id is prohibited.
    for version, entry in versions.items():
        if entry.get("model_id", version) in versions and entry.get("model_id", version) != version:
            raise DataValidationError("Model ID conflicts with another version alias")
    return config


class ModelRegistry:
    def __init__(self, root, config_path="configs/model_versions.json"):
        self.root = Path(root).resolve()
        self.config = load_version_registry(self.root, config_path)

    @property
    def default_version(self):
        return self.config["default_version"]

    def resolve(self, version_or_id=None):
        requested = self.default_version if version_or_id is None else version_or_id
        if requested in self.config["versions"]:
            return requested, self.config["versions"][requested]
        for version, entry in self.config["versions"].items():
            if entry.get("model_id", version) == requested:
                return version, entry
        raise DataValidationError(f"Unknown model_version/model ID: {requested}; available: {', '.join(self.config['versions'])}")

    def is_available(self, version_or_id=None):
        _, entry = self.resolve(version_or_id)
        path = self.root / entry["model_dir"]
        return all((path / name).is_file() for name in BUNDLE_FILES)

    def get_metadata(self, version_or_id=None):
        _, entry = self.resolve(version_or_id)
        return read_model_metadata(self.root / entry["model_dir"])

    def get(self, version_or_id=None):
        version, entry = self.resolve(version_or_id)
        available = self.is_available(version)
        metadata = self.get_metadata(version) if available else None
        return ModelRecord(ModelIdentity(entry.get("model_id", version), version, entry.get("model_type", "als"),
            metadata.get("train_fingerprint") if metadata else None,
            metadata.get("mapping_fingerprint") if metadata else None),
            self.root / entry["model_dir"], available, metadata)

    def list_models(self):
        return [self.get(version) for version in self.config["versions"]]

    def load(self, version_or_id=None, *, expected_mapping=None, expected_train_fingerprint=None):
        version, entry = self.resolve(version_or_id)
        if not self.is_available(version):
            raise DataValidationError(f"Saved model {version} is missing/incomplete; prepare it separately. No automatic fallback.")
        configured = entry.get("train_fingerprint")
        if configured is not None and expected_train_fingerprint is not None and configured != expected_train_fingerprint:
            raise DataValidationError("Registry/caller training fingerprints disagree")
        bundle = load_models(self.root / entry["model_dir"], expected_mapping,
                             expected_train_fingerprint if expected_train_fingerprint is not None else configured)
        if entry.get("mapping_fingerprint") is not None and entry["mapping_fingerprint"] != bundle.metadata["mapping_fingerprint"]:
            raise DataValidationError("Registry/model mapping fingerprint mismatch")
        # Legacy v1 has no logical version: config supplies it without rewriting metadata.
        if (bundle.metadata.get("model_version") is not None and
                bundle.metadata["model_version"] != entry.get("artifact_version", version)):
            raise DataValidationError("Saved artifact version differs from selected model_version")
        return bundle

    def verify(self, version_or_id=None, **expected):
        self.load(version_or_id, **expected)
        return self.get(version_or_id)
