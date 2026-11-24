"""Compatibility API for the explicitly combined ARE-008 research experiment."""
from anime_recommender.training.fitting import fit_models
from anime_recommender.evaluation.scaling_overview import (
    fixed_validation_identity, check_comparison_compatibility, load_comparison, selective_scaling_overview,
)

def run_comparison(root, *, train_new=False, validation_user_id=None):
    from anime_recommender.pipelines.legacy_comparison import run_comparison as call
    return call(root, train_new=train_new, validation_user_id=validation_user_id, fitter=fit_models)

def add_baseline_deltas(frame):
    from anime_recommender.pipelines.legacy_comparison import add_baseline_deltas as call
    return call(frame)

def comparison_row(*args, **kwargs):
    from anime_recommender.pipelines.legacy_comparison import comparison_row as call
    return call(*args, **kwargs)

def main():
    from anime_recommender.pipelines.legacy_comparison import main as call
    return call()

if __name__ == "__main__":
    main()
