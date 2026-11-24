"""Legacy ARE-005 data-preparation API; no model training."""
from anime_recommender.preprocessing.dataset import (
    PreparedDataset, validate_config, prepare_dataset, run_pipeline, load_dataset, main,
)

if __name__ == "__main__":
    main()
