"""Stable train row/column mappings and sparse binary confidence matrix."""

import pandas as pd
from scipy.sparse import csr_matrix

from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids, require_columns


def build_train_matrix(train: pd.DataFrame, catalog_mapping: pd.DataFrame):
    require_columns(train, ["dataset_user_id", "dataset_anime_id", "confidence"], "train")
    require_columns(catalog_mapping, ["dataset_anime_id", "myanimelist_id"], "catalog mapping")
    if train.duplicated(["dataset_user_id", "dataset_anime_id"]).any():
        raise DataValidationError("Duplicate train pairs would sum sparse confidence")
    mapping = catalog_mapping[["dataset_anime_id", "myanimelist_id"]].copy()
    for col in mapping:
        mapping[col] = positive_ids(mapping[col], col)
        if mapping[col].duplicated().any():
            raise DataValidationError(f"Catalog mapping has ambiguous {col}")
    users = pd.DataFrame({"dataset_user_id": pd.Series(sorted(train["dataset_user_id"].unique()), dtype="int64")})
    users["matrix_row"] = range(len(users))
    items = pd.DataFrame({"dataset_anime_id": pd.Series(sorted(train["dataset_anime_id"].unique()), dtype="int64")})
    items["matrix_col"] = range(len(items))
    items = items.merge(mapping, on="dataset_anime_id", how="left", validate="one_to_one")
    if items["myanimelist_id"].isna().any():
        raise DataValidationError("Train item missing an unambiguous MAL mapping")
    row = train["dataset_user_id"].map(users.set_index("dataset_user_id")["matrix_row"])
    col = train["dataset_anime_id"].map(items.set_index("dataset_anime_id")["matrix_col"])
    matrix = csr_matrix((train["confidence"].to_numpy(dtype="float32"),
                         (row.to_numpy(dtype="int64"), col.to_numpy(dtype="int64"))),
                         shape=(len(users), len(items)), dtype="float32")
    matrix.sort_indices()
    return matrix, users, items


def build_context_matrix(context: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    """Only warm positive context; no target argument, all cohort users get rows."""
    known = context["dataset_anime_id"].isin(items["dataset_anime_id"])
    if not context["is_cold_item"].eq(~known).all():
        raise DataValidationError("Context cold flags disagree with train vocabulary")
    warm = context.loc[known]
    if warm.duplicated(["dataset_user_id", "dataset_anime_id"]).any():
        raise DataValidationError("Duplicate warm context pairs")
    row = warm["dataset_user_id"].map(users.set_index("dataset_user_id")["matrix_row"])
    col = warm["dataset_anime_id"].map(items.set_index("dataset_anime_id")["matrix_col"])
    if row.isna().any() or col.isna().any():
        raise DataValidationError("Context mapping does not cover its warm entries")
    matrix = csr_matrix((warm["confidence"].to_numpy(dtype="float32"),
        (row.to_numpy(dtype="int64"), col.to_numpy(dtype="int64"))),
        shape=(len(users), len(items)), dtype="float32")
    matrix.sort_indices()
    return matrix
