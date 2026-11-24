"""Load-only canonical validation evaluation, optional new report publication."""
from pathlib import Path
from tempfile import TemporaryDirectory

from anime_recommender.artifact_files import publish_directory
from anime_recommender.artifact_io import load_models
from anime_recommender.evaluation.ranking_evaluator import evaluate_models
from anime_recommender.evaluation.reports import load_report, save_report
from anime_recommender.validation_inputs import load_validation_view


def run_evaluation(model_dir, validation_dir, config, *, output_dir=None):
    output = Path(output_dir) if output_dir is not None else None
    if output is not None and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"Evaluation report already exists: {output}")
    if set(config) - {"ks", "include_als_fallback", "split"} or config.get("split", "validation") != "validation":
        raise ValueError("Only canonical validation evaluation is supported")
    bundle = load_models(model_dir)
    inputs = load_validation_view(validation_dir, bundle)
    report = evaluate_models(bundle, inputs, config.get("ks", [10]),
                             config.get("include_als_fallback", False))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".evaluation-", dir=output.parent) as temporary:
            staging = Path(temporary)
            save_report(report, staging)
            load_report(staging)
            publish_directory(staging, output)
    return report
