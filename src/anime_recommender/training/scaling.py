"""Legacy scaling preparation entrypoint; implementation belongs to data preparation."""
import argparse
import json
from pathlib import Path
from anime_recommender.preprocessing.scaling import (
    PROTOCOL, _same, fingerprint, nested_users, prepare_scaling, save_scaling_dataset,
    load_scaling_dataset, register_baseline, scaling_summary, synthetic_scaling_demo,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaling.json")
    parser.add_argument("--execute", action="store_true", help="Prepare from verified local caches only")
    parser.add_argument("--allow-full-scan", action="store_true", help="Explicit approval for one 10k source scan")
    args = parser.parse_args()
    from anime_recommender.pipelines.scaling import run_scaling
    print(json.dumps(run_scaling(Path.cwd(), args.config, execute=args.execute,
                                 allow_full_scan=args.allow_full_scan), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Scaling preparation failed: {error}") from None
