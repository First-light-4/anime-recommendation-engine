"""Load-only certified catalog/model workspace, no dataset interaction reads."""
import json
from pathlib import Path
import pandas as pd
from anime_recommender.artifact_io import load_models
from anime_recommender.artifact_files import file_sha256
from anime_recommender.contracts import ModelWorkspace, DataValidationError
from anime_recommender.preprocessing.franchise_mapping import load_franchise_resources

def load_profile_workspace(root, model_dir=None, *, model_loader=load_models):
    """Load only saved models and the certified catalog, with no training/splits.

    A missing bundle never invokes fit or downloads anything. The preparation
    instruction deliberately points to the separate existing training pipeline.
    """
    root = Path(root)
    config = json.loads((root / "configs/recommendation_models.json").read_text(encoding="utf-8"))
    if model_dir is not None:
        config["model_dir"] = str(model_dir)
    try:
        bundle = model_loader(root / config["model_dir"])
        path = root / config["catalog_path"]
        report = json.loads((path.parent / "data_quality_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "success" or file_sha256(path) != report.get("output_sha256", {}).get(path.name):
            raise DataValidationError("Display catalog is not certified by ARE-004")
        catalog = pd.read_parquet(path)
        aligned = bundle.items[["dataset_anime_id", "myanimelist_id"]].merge(
            catalog[["dataset_anime_id", "myanimelist_id"]], on="dataset_anime_id",
            how="left", validate="one_to_one", suffixes=("_model", "_catalog"))
        if not aligned.myanimelist_id_model.eq(aligned.myanimelist_id_catalog).all():
            raise DataValidationError("Catalog MAL mapping differs from saved model")
    except FileNotFoundError:
        raise DataValidationError(
            "Saved models/catalog are missing. Prepare local ARE-005.1 data, then run "
            "python -m anime_recommender.training.pipeline from the project root. "
            "Notebook 05 only loads models and never trains them.") from None
    franchise_config, franchise_mapping = load_franchise_resources(root)
    return ModelWorkspace(bundle, None, catalog, franchise_config, franchise_mapping)
