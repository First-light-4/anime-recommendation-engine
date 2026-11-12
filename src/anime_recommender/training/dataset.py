"""Prepare ML artifacts from local ARE-004 tables. No source/API calls or ALS."""

import argparse
from dataclasses import dataclass, field
from importlib.metadata import version
import json
import math
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from anime_recommender.ingestion.processed_data import load_processed_data
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.preprocessing.implicit_feedback import positive_feedback, threshold_summary
from anime_recommender.preprocessing.data_split import assign_users, eligible_positives, filter_train, split_observed_interactions
from anime_recommender.preprocessing.sparse_matrix import build_train_matrix, build_context_matrix
from anime_recommender.evaluation.dataset_validation import evaluation_summary, evaluation_user_mapping, validate_dataset, validate_context_matrix


def validate_config(config, root):
    defaults = {"input_dir", "output_dir", "seed", "positive_threshold", "feedback", "confidence",
                "train_fraction", "validation_fraction", "test_fraction", "min_user_positives",
                "min_item_users", "context_fraction", "min_context", "min_target", "evaluation_protocol"}
    if not isinstance(config, dict) or not defaults <= config.keys():
        raise DataValidationError("ML config is missing required parameters")
    for key in ["seed", "positive_threshold", "min_user_positives", "min_item_users", "min_context", "min_target"]:
        if type(config[key]) is not int or (key != "seed" and config[key] < 1):
            raise DataValidationError(f"Invalid integer parameter: {key}")
    if config["positive_threshold"] not in (5, 6, 7, 8) or config["feedback"] != "binary" or config["confidence"] != 1.0 or isinstance(config["confidence"], bool):
        raise DataValidationError("Use threshold 5/6/7/8, binary feedback, confidence=1")
    fractions = [config[key] for key in ["train_fraction", "validation_fraction", "test_fraction"]]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in fractions) or not math.isclose(sum(fractions), 1):
        raise DataValidationError("Split fractions must be positive and sum to one")
    ratio = config["context_fraction"]
    if type(ratio) not in (int, float) or not 0 < ratio < 1:
        raise DataValidationError("context_fraction must be between zero and one")
    if config["evaluation_protocol"] != "raw_holdout_then_positive_v2":
        raise DataValidationError("Use raw_holdout_then_positive_v2 evaluation protocol")
    for key in ["input_dir", "output_dir"]:
        if not isinstance(config[key], str) or Path(config[key]).is_absolute() or not (root / config[key]).resolve().is_relative_to((root / "data").resolve()):
            raise DataValidationError(f"{key} must be a relative path inside data/")
    if (root / config["output_dir"]).resolve() == (root / config["input_dir"]).resolve():
        raise DataValidationError("ML output_dir must differ from ARE-004 input_dir")


@dataclass
class PreparedDataset:
    tables: dict[str, pd.DataFrame]
    matrix: csr_matrix
    report: dict
    context_matrices: dict[str, csr_matrix] = field(default_factory=dict)


def prepare_dataset(interactions, catalog_mapping, config) -> PreparedDataset:
    # No rating threshold or evaluation activity selection before the raw holdout.
    assignments = assign_users(interactions["dataset_user_id"], config["seed"],
        tuple(config[key] for key in ["train_fraction", "validation_fraction", "test_fraction"]))
    allocated = assignments["split"].value_counts().to_dict()
    if any(allocated.get(name, 0) == 0 for name in ("train", "validation", "test")):
        raise DataValidationError("Not enough users to allocate three nonempty cohorts; thresholds unchanged")
    raw = {name: interactions.loc[interactions["dataset_user_id"].isin(
        assignments.loc[assignments["split"].eq(name), "dataset_user_id"])].copy()
        for name in ("train", "validation", "test")}
    raw_evaluations = {name: split_observed_interactions(raw[name], config["seed"],
        config["context_fraction"], config["min_context"], config["min_target"])
        for name in ("validation", "test")}
    threshold = config["positive_threshold"]
    train_positive = positive_feedback(raw["train"], threshold)
    train_candidates = eligible_positives(train_positive, assignments, "train", config["min_user_positives"])
    train, rounds = filter_train(train_candidates, config["min_user_positives"], config["min_item_users"])
    if train.empty:
        raise DataValidationError("Train is empty after configured train-only k-core; thresholds unchanged")
    matrix, train_users, items = build_train_matrix(train, catalog_mapping)
    evaluations = {}
    eval_users = {}
    context_matrices = {}
    for name, (raw_context, raw_target) in raw_evaluations.items():
        context, target = (positive_feedback(frame, threshold) for frame in (raw_context, raw_target))
        for frame in (context, target):
            frame["is_cold_item"] = ~frame["dataset_anime_id"].isin(items["dataset_anime_id"])
        evaluations[name] = (context, target)
        cohort = assignments.loc[assignments["split"].eq(name), "dataset_user_id"]
        eval_users[name] = evaluation_user_mapping(cohort, raw_context, raw_target, context, target)
        context_matrices[name] = build_context_matrix(context, eval_users[name], items)
    checks = validate_dataset(train, evaluations, assignments, train_users, items, matrix,
        {name: positive_feedback(raw[name], threshold) for name in evaluations},
        config["min_user_positives"], config["min_item_users"], raw_evaluations,
        {name: raw[name] for name in evaluations}, threshold)
    positives = positive_feedback(interactions, threshold)  # diagnostics only, after holdout
    counts = positives.groupby("dataset_user_id").size()
    assignments["positive_count"] = assignments["dataset_user_id"].map(counts).fillna(0).astype("int64")
    assignments["included"] = ~assignments["split"].eq("train") | assignments["dataset_user_id"].isin(train_users["dataset_user_id"])
    tables = {"train_interactions": train, "user_assignments": assignments,
              "train_user_mapping": train_users, "item_mapping": items,
              "threshold_comparison": threshold_summary(interactions)}
    warnings = []
    summaries = {}
    for name, (context, target) in evaluations.items():
        tables[name + "_context"] = context
        tables[name + "_target"] = target
        tables[name + "_observed_context"] = raw_evaluations[name][0]
        tables[name + "_observed_target"] = raw_evaluations[name][1]
        tables[name + "_user_mapping"] = eval_users[name]
        cohort = set(assignments.loc[assignments["split"].eq(name), "dataset_user_id"])
        if set(eval_users[name]["dataset_user_id"]) != cohort:
            raise DataValidationError("Evaluation user was dropped")
        csr = context_matrices[name]
        validate_context_matrix(context, eval_users[name], items, csr)
        checks[name + "_all_assigned_users_retained"] = True
        checks[name + "_context_csr_warm_only"] = True
        summary = evaluation_summary(context, target, cohort)
        summary["observed_context_rows"] = len(raw_evaluations[name][0])
        summary["observed_target_rows"] = len(raw_evaluations[name][1])
        summary["context_csr"] = {"shape": list(csr.shape), "nnz": csr.nnz}
        summaries[name] = summary
        if summary["users_without_known_context"]:
            warnings.append(f"{name}: {summary['users_without_known_context']} users without warm positive context retained")
        if summary["users_without_positive_targets"]:
            warnings.append(f"{name}: {summary['users_without_positive_targets']} users without positive targets retained; Recall undefined")
    report = {"status": "success", "dataset_version": 2, "config": config,
        "input": {"rows": len(interactions), "users": int(interactions["dataset_user_id"].nunique()),
                  "positive_rows": len(positives), "positive_users": int(positives["dataset_user_id"].nunique()),
                  "nonpositive_or_missing_rows": len(interactions) - len(positives)},
        "cohorts": {name: {"allocated_users": allocated.get(name, 0),
            "retained_users": len(train_users) if name == "train" else len(eval_users[name]),
            "positive_rows_before_train_filters": len(train_positive) if name == "train" else summaries[name]["positive_rows"],
            "evaluation_activity_filter_applied": False} for name in raw},
        "train": {"users": len(train_users), "anime": len(items), "positive_rows": len(train),
            "activity_excluded_users": allocated["train"] - int(train_candidates["dataset_user_id"].nunique()),
            "activity_excluded_positive_rows": len(train_positive) - len(train_candidates),
            "filtered_rows": len(train_candidates) - len(train),
            "filtered_users": int(train_candidates["dataset_user_id"].nunique()) - len(train_users),
            "filter_rounds": rounds},
        "evaluation": summaries,
        "matrix": {"shape": list(matrix.shape), "nnz": matrix.nnz,
            "dtype": str(matrix.dtype), "sparsity": 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])},
        "checks": checks, "warnings": warnings,
        "protocol": "User-disjoint; raw history holdout before positive feedback; all eval users and cold targets retained"}
    return PreparedDataset(tables, matrix, report, context_matrices)


def run_pipeline(config, root) -> PreparedDataset:
    root = Path(root).resolve()
    validate_config(config, root)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        data = load_processed_data(root / config["input_dir"])
        prepared = prepare_dataset(data.interactions, data.mapping, config)
        for name, frame in prepared.tables.items():
            atomic_parquet(frame, output / (name + ".parquet"))
        temporary = output / "train_matrix.tmp.npz"
        save_npz(temporary, prepared.matrix, compressed=True)
        temporary.replace(output / "train_matrix.npz")
        for name, context_matrix in prepared.context_matrices.items():
            temporary = output / (name + "_context_matrix.tmp.npz")
            save_npz(temporary, context_matrix, compressed=True)
            temporary.replace(output / (name + "_context_matrix.npz"))
        atomic_json(output / "config.json", config)
        prepared.report["source_sha256"] = data.quality_report["output_sha256"]
        prepared.report["environment"] = {name: version(name) for name in ("pandas", "scipy", "pyarrow")}
        names = [name + ".parquet" for name in prepared.tables] + ["train_matrix.npz", "config.json"] + [name + "_context_matrix.npz" for name in prepared.context_matrices]
        prepared.report["artifact_sha256"] = {name: file_sha256(output / name) for name in names}
        atomic_json(output / "dataset_report.json", prepared.report)
        return prepared
    except (OSError, ValueError) as error:
        atomic_json(output / "dataset_report.json", {"status": "failed", "config": config,
            "error": str(error), "note": "Old output files, if present, are not certified"})
        raise


def load_dataset(directory, expected_config=None, expected_source_sha256=None) -> PreparedDataset:
    directory = Path(directory)
    report = json.loads((directory / "dataset_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise DataValidationError("ML dataset preparation did not succeed")
    if report.get("dataset_version") != 2:
        raise DataValidationError("Legacy ML dataset version; rebuild with the corrected v2 protocol")
    if expected_config is not None and report.get("config") != expected_config:
        raise DataValidationError("Saved ML configuration differs; rebuild the local ML artifacts")
    if expected_source_sha256 is not None and report.get("source_sha256") != expected_source_sha256:
        raise DataValidationError("Saved ML source hashes differ; rebuild the local ML artifacts")
    for name, expected in report["artifact_sha256"].items():
        if not (directory / name).is_file() or file_sha256(directory / name) != expected:
            raise DataValidationError(f"ML artifact checksum mismatch: {name}")
    tables = {Path(name).stem: pd.read_parquet(directory / name)
              for name in report["artifact_sha256"] if name.endswith(".parquet")}
    matrices = {name: load_npz(directory / (name + "_context_matrix.npz")) for name in ("validation", "test")}
    return PreparedDataset(tables, load_npz(directory / "train_matrix.npz"), report, matrices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml_dataset.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    print(json.dumps(run_pipeline(config, root).report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ML dataset preparation failed: {error}") from None
