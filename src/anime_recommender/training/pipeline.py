"""Legacy explicitly combined ARE-006 experiment API. Prefer pipelines.training."""
from anime_recommender.training.fitting import fit_models

def __getattr__(name):
    if name == "Experiment":
        from anime_recommender.pipelines.legacy_experiment import Experiment
        return Experiment
    raise AttributeError(name)

def run_experiment(config, root, train=True):
    from anime_recommender.pipelines.legacy_experiment import run_experiment as call
    return call(config, root, train)

def validate_config(config, root):
    from anime_recommender.pipelines.legacy_experiment import validate_config as call
    return call(config, root)

def main():
    from anime_recommender.pipelines.legacy_experiment import main as call
    return call()

if __name__ == "__main__":
    main()
