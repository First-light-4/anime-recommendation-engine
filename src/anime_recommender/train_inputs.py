"""Read-only train view; no validation/test dependency or automatic preparation."""
from dataclasses import dataclass
import pandas as pd
from scipy.sparse import load_npz, isspmatrix_csr

from anime_recommender.contracts import DataValidationError, validate_item_mapping
from anime_recommender.dataset_artifacts import (
    TRAIN_FILES, dataset_manifest, read_tables, training_identity, verified_path,
)


@dataclass
class TrainView:
    train: pd.DataFrame
    train_users: pd.DataFrame
    items: pd.DataFrame
    matrix: object
    dataset_config: dict
    train_fingerprint: str
    train_hashes: dict
    catalog_sha256: str


def load_train_view(directory) -> TrainView:
    directory, report = dataset_manifest(directory)
    tables = read_tables(directory, report, TRAIN_FILES[:3])
    matrix = load_npz(verified_path(directory, report, "train_matrix.npz"))
    train, users, items = (tables[name] for name in TRAIN_FILES[:3])
    validate_item_mapping(items)
    if (not isspmatrix_csr(matrix) or matrix.shape != (len(users), len(items))
            or matrix.nnz != len(train)):
        raise DataValidationError("Train CSR and mappings do not agree")
    fingerprint, hashes = training_identity(report)
    return TrainView(train, users, items, matrix, report["config"], fingerprint, hashes,
                     report["source_sha256"]["anime_catalog.parquet"])
