"""Fit CPU implicit ALS on users-by-items CSR; no evaluation inputs."""

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import isspmatrix_csr
from threadpoolctl import threadpool_limits

from anime_recommender.preprocessing.training_data import DataValidationError


def validate_factors(model, users, items):
    if (model.user_factors.shape != (users, model.factors)
            or model.item_factors.shape != (items, model.factors)
            or not np.isfinite(model.user_factors).all() or not np.isfinite(model.item_factors).all()):
        raise DataValidationError("ALS factor orientation, dimensions or finite values are invalid")


def train_als(train_matrix, config, blas_threads=1):
    if not isspmatrix_csr(train_matrix) or not all(train_matrix.shape) or train_matrix.nnz == 0:
        raise DataValidationError("ALS requires a nonempty users-by-items CSR")
    if not np.isfinite(train_matrix.data).all() or not (train_matrix.data > 0).all():
        raise DataValidationError("ALS requires finite positive confidences")
    expected = {"factors", "regularization", "iterations", "random_state", "num_threads", "alpha", "dtype", "use_cg", "cg_steps"}
    if set(config) != expected:
        raise DataValidationError("Unexpected/missing ALS config parameters")
    for key in ("factors", "iterations", "num_threads", "cg_steps"):
        if type(config[key]) is not int or config[key] <= 0:
            raise DataValidationError(f"ALS {key} must be a positive integer")
    if (type(config["random_state"]) is not int or config["regularization"] <= 0
            or not np.isfinite(config["regularization"]) or config["alpha"] != 1
            or config["dtype"] != "float32" or type(config["use_cg"]) is not bool):
        raise DataValidationError("Invalid ALS config; keep binary confidence alpha=1, float32 and positive regularization")
    if type(blas_threads) is not int or blas_threads < 1:
        raise DataValidationError("blas_threads must be positive")
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        model = AlternatingLeastSquares(**{k: v for k, v in config.items() if k != "cg_steps"})
        model.cg_steps = config["cg_steps"]
        # implicit 0.7.3 accepts users x items directly, not its transpose.
        model.fit(train_matrix, show_progress=False)
    validate_factors(model, *train_matrix.shape)
    return model
