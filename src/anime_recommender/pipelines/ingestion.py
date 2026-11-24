"""Source acquisition only. No cleaning, splits, models or evaluation."""
from pathlib import Path

from anime_recommender.ingestion.config import validate_config
from anime_recommender.ingestion.training_data import acquire, load_cached_raw


def run_ingestion(config, root, *, execute=False, allow_network=False):
    root = Path(root).resolve()
    validate_config(config, root)
    cache = root / config["cache_dir"]
    if (cache / "manifest.json").is_file():
        _, _, manifest = load_cached_raw(config, root)
        return {"mode": "reused", "raw_dir": str(cache), "manifest": manifest}
    if not execute:
        return {"mode": "plan", "raw_dir": str(cache), "network_required": True,
                "source_bytes": config["source"]["ratings_bytes"]}
    if not allow_network:
        raise ValueError("Source acquisition requires explicit allow_network=True")
    _, _, manifest = acquire(config, root, allow_full_scan=True)
    return {"mode": "acquired", "raw_dir": str(cache), "manifest": manifest}
