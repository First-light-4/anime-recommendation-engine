"""Selective ARE-005.1 reader: train and validation only, never opens test artifacts."""

from dataclasses import dataclass

import pandas as pd



@dataclass
class TrainingInputs:
    train: pd.DataFrame
    train_users: pd.DataFrame
    items: pd.DataFrame
    matrix: object
    validation_context: pd.DataFrame
    validation_target: pd.DataFrame
    observed_context: pd.DataFrame
    validation_users: pd.DataFrame
    dataset_config: dict
    train_fingerprint: str
    train_hashes: dict
    catalog_sha256: str


def load_training_inputs(directory) -> TrainingInputs:
    """Legacy combined view. New runners use selective readers directly."""
    from types import SimpleNamespace
    from anime_recommender.train_inputs import load_train_view
    from anime_recommender.validation_inputs import load_validation_view
    from anime_recommender.artifact_io import mapping_fingerprint
    train = load_train_view(directory)
    identity = SimpleNamespace(items=train.items, train_users=train.train_users, metadata={
        "train_fingerprint": train.train_fingerprint,
        "mapping_fingerprint": mapping_fingerprint(train.items)})
    validation = load_validation_view(directory, identity)
    return TrainingInputs(train.train, train.train_users, train.items, train.matrix,
        validation.validation_context, validation.validation_target, validation.observed_context,
        validation.validation_users, train.dataset_config, train.train_fingerprint,
        train.train_hashes, train.catalog_sha256)
