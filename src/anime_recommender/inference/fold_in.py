"""Exact fixed-item ALS user solve through implicit.recalculate_user."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from anime_recommender.inference.recommendations import context_ids, validate_item_mapping
from anime_recommender.preprocessing.training_data import DataValidationError


@dataclass
class FoldInResult:
    available: bool
    factor: np.ndarray | None
    context_vector: csr_matrix
    diagnostics: dict


def fold_in(model, user_context, item_mapping):
    mapping = validate_item_mapping(item_mapping)
    if model.item_factors.shape != (len(mapping), model.factors):
        raise DataValidationError("Item mapping does not match ALS factor dimensions")
    ids = context_ids(user_context)
    columns = mapping.set_index("dataset_anime_id")["matrix_col"].to_dict()
    warm = [columns[item] for item in ids if item in columns]
    vector = csr_matrix((np.ones(len(warm), dtype="float32"),
                         (np.zeros(len(warm), dtype="int32"), np.array(warm, dtype="int32"))),
                         shape=(1, len(mapping)), dtype="float32")
    diagnostics = {"context_unique": len(ids), "warm_context": len(warm), "cold_context": len(ids) - len(warm)}
    if not warm:
        return FoldInResult(False, None, vector, diagnostics)
    with threadpool_limits(limits=1, user_api="blas"):
        # This solves a regularized least-squares problem with fixed item factors.
        # It does not store the factor under a train user ID or call fit/partial_fit.
        factor = model.recalculate_user(0, vector)
    if factor.shape != (model.factors,) or not np.isfinite(factor).all():
        raise DataValidationError("Fold-in returned invalid user factors")
    return FoldInResult(True, factor, vector, diagnostics)
