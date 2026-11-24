"""Train-only ALS/Popularity fitting. No validation, evaluation or ingestion."""
from importlib.metadata import version
from time import perf_counter
from anime_recommender.training.als import train_als
from anime_recommender.training.popularity import train_popularity
from anime_recommender.inference.bundle import ModelBundle

def fit_models(inputs, config):
    start = perf_counter()
    popularity = train_popularity(inputs.train, inputs.items)
    pop_seconds = perf_counter() - start
    start = perf_counter()
    als = train_als(inputs.matrix, config["als"], config["blas_threads"])
    als_seconds = perf_counter() - start
    metadata = {"format_version": 1, "training_config": config,
        "dataset_config": inputs.dataset_config, "train_fingerprint": inputs.train_fingerprint,
        "train_hashes": inputs.train_hashes, "matrix_orientation": "users_by_items",
        "train_shape": list(inputs.matrix.shape), "train_nnz": inputs.matrix.nnz,
        "fit_seconds": {"popularity": pop_seconds, "als": als_seconds},
        "factor_shapes": {"users": list(als.user_factors.shape), "items": list(als.item_factors.shape)},
        "versions": {name: version(name) for name in ("implicit", "numpy", "scipy", "pandas", "pyarrow", "threadpoolctl")},
        "confidence_policy": "binary confidence=1, implicit alpha=1; no tuning",
        "training_split": "train only"}
    return ModelBundle(als, popularity, inputs.items.copy(), inputs.train_users.copy(), metadata)
