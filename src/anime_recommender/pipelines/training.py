"""Explicit train-only operation; publication and fitting require execute=True."""
from pathlib import Path

from anime_recommender.artifact_io import save_models
from anime_recommender.train_inputs import load_train_view
from anime_recommender.training.fitting import fit_models


def run_training(dataset_dir, config, output_dir, *, execute=False):
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Model bundle already exists: {output}")
    inputs = load_train_view(dataset_dir)
    if not execute:
        return {"execute": False, "train_shape": list(inputs.matrix.shape),
                "train_nnz": inputs.matrix.nnz, "train_fingerprint": inputs.train_fingerprint,
                "output_dir": str(output)}
    bundle = fit_models(inputs, config)
    save_models(bundle, output)
    return bundle
