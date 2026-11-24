"""Legacy explicit combined scaling workflow; use separate stages for new work."""
from anime_recommender.ingestion.training_data import acquire
from anime_recommender.pipelines.scaling import run_scaling as _run_scaling

def run_scaling(root, config_path="configs/scaling.json", *, execute=False, allow_full_scan=False):
    return _run_scaling(root, config_path, execute=execute,
                        allow_full_scan=allow_full_scan, acquirer=acquire)
