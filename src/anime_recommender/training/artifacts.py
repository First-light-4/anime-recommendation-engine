"""Safe numeric NPZ/Parquet model bundle with explicit mapping/data fingerprints."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from implicit.cpu.als import AlternatingLeastSquares
import pandas as pd
from threadpoolctl import threadpool_limits

from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.inference.als import ALSRecommender
from anime_recommender.inference.popularity import PopularityRecommender
from anime_recommender.inference.recommendations import validate_item_mapping
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.training.als import validate_factors


def mapping_fingerprint(mapping):
    values = validate_item_mapping(mapping).to_numpy(dtype="int64").tolist()
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


@dataclass
class ModelBundle:
    als: object
    popularity: pd.DataFrame
    items: pd.DataFrame
    train_users: pd.DataFrame
    metadata: dict

    def recommenders(self):
        return {"popularity": PopularityRecommender(self.popularity, self.items),
                "als": ALSRecommender(self.als, self.items)}


def save_models(bundle: ModelBundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "als.tmp.npz"
    bundle.als.save(temporary)
    temporary.replace(directory / "als.npz")
    atomic_parquet(bundle.items, directory / "item_mapping.parquet")
    atomic_parquet(bundle.train_users, directory / "train_user_mapping.parquet")
    atomic_parquet(bundle.popularity, directory / "popularity.parquet")
    atomic_json(directory / "training_config.json", bundle.metadata["training_config"])
    bundle.metadata["mapping_fingerprint"] = mapping_fingerprint(bundle.items)
    bundle.metadata["files_sha256"] = {name: file_sha256(directory / name) for name in (
        "als.npz", "item_mapping.parquet", "train_user_mapping.parquet", "popularity.parquet", "training_config.json")}
    atomic_json(directory / "metadata.json", bundle.metadata)


def load_models(directory, expected_mapping=None, expected_train_fingerprint=None) -> ModelBundle:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected_files = {"als.npz", "item_mapping.parquet", "train_user_mapping.parquet", "popularity.parquet", "training_config.json"}
    if set(metadata.get("files_sha256", {})) != expected_files:
        raise DataValidationError("Unexpected model bundle file list")
    for name, sha in metadata["files_sha256"].items():
        if file_sha256(directory / name) != sha:
            raise DataValidationError(f"Model checksum mismatch: {name}")
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
