"""Local integrity/coverage audit for scaling artifacts, without model evaluation."""

import json
from pathlib import Path

import pandas as pd

from anime_recommender.preprocessing.dataset_validation import validate_context_matrix, validate_dataset
from anime_recommender.artifact_files import atomic_json, file_sha256
from anime_recommender.preprocessing.implicit_feedback import positive_feedback
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.preprocessing.scaling import _same, scaling_summary


def audit_scaling_datasets(datasets, *, protected_sha256=None, root=None):
    """Return aggregate-only tables; targets serve integrity/coverage checks only."""
    if not datasets:
        raise DataValidationError("No scaling datasets to audit")
    baseline = next(iter(datasets.values()))
    baseline_assignments = baseline.tables["user_assignments"]
    previous_users, previous_pairs = set(), set()
    coverage, matrices, activity = [], [], []
    for name, data in datasets.items():
        tables, report = data.tables, data.report
        train = tables["train_interactions"]
        assignments = tables["user_assignments"]
        users = set(tables["train_user_mapping"].dataset_user_id)
        pairs = set(zip(train.dataset_user_id, train.dataset_anime_id))
        if not previous_users <= users or not previous_pairs <= pairs:
            raise DataValidationError("Scaling train populations/interactions are not nested")
        allocated = assignments.loc[assignments.dataset_user_id.isin(baseline_assignments.dataset_user_id),
                                    ["dataset_user_id", "split"]].sort_values("dataset_user_id")
        _same(allocated, baseline_assignments[["dataset_user_id", "split"]].sort_values("dataset_user_id"),
              "baseline assignments")
        if report["config"]["positive_threshold"] != 5:
            raise DataValidationError("Scaling threshold changed")
        raw = {cohort: tuple(tables[f"{cohort}_observed_{part}"] for part in ("context", "target"))
               for cohort in ("validation", "test")}
        evaluations = {cohort: tuple(tables[f"{cohort}_{part}"] for part in ("context", "target"))
                       for cohort in raw}
        raw_cohorts = {cohort: pd.concat(parts, ignore_index=True) for cohort, parts in raw.items()}
        validate_dataset(train, evaluations, assignments, tables["train_user_mapping"],
            tables["item_mapping"], data.matrix,
            {cohort: positive_feedback(frame, 5) for cohort, frame in raw_cohorts.items()},
            5, 5, raw, raw_cohorts, 5)
        matrices.append({"sample": name, "matrix": "train", "rows": data.matrix.shape[0],
                         "columns": data.matrix.shape[1], "nnz": data.matrix.nnz})
        for cohort in raw:
            mapping = tables[cohort + "_user_mapping"]
            expected_ids = baseline.tables[cohort + "_user_mapping"].dataset_user_id
            if set(mapping.dataset_user_id) != set(expected_ids):
                raise DataValidationError("Fixed evaluation user cohort changed")
            validate_context_matrix(evaluations[cohort][0], mapping, tables["item_mapping"],
                                    data.context_matrices[cohort])
            matrix = data.context_matrices[cohort]
            matrices.append({"sample": name, "matrix": cohort + "_context", "rows": matrix.shape[0],
                             "columns": matrix.shape[1], "nnz": matrix.nnz})
            for part in ("context", "target", "observed_context", "observed_target"):
                key = cohort + "_" + part
                frame = tables[key]
                _same(frame.drop(columns="is_cold_item", errors="ignore"),
                      baseline.tables[key].drop(columns="is_cold_item", errors="ignore"), key)
                if part in ("context", "target"):
                    cold = int(frame.is_cold_item.sum())
                    coverage.append({"sample": name, "cohort": cohort, "part": part, "rows": len(frame),
                        "warm_rows": len(frame) - cold, "cold_rows": cold,
                        "warm_share": (len(frame) - cold) / len(frame) if len(frame) else None,
                        "cold_share": cold / len(frame) if len(frame) else None})
        quality = report.get("cleaning")
        if quality is not None and quality["input_rows"] != quality["output_rows"] + sum(quality["removed_by_reason"].values()):
            raise DataValidationError("Cleaning removal accounting mismatch")
        train_report = report["train"]
        before = report["cohorts"]["train"]["positive_rows_before_train_filters"]
        if before != len(train) + train_report["activity_excluded_positive_rows"] + train_report["filtered_rows"]:
            raise DataValidationError("Train filter accounting mismatch")
        if report["input"]["positive_rows"] != before + sum(len(c) + len(t) for c, t in evaluations.values()):
            raise DataValidationError("Positive cohort accounting mismatch")
        counts = train.groupby("dataset_user_id").size()
        activity.append({"sample": name, "min": int(counts.min()), "median": float(counts.median()),
            "mean": float(counts.mean()), "p90": float(counts.quantile(.9)), "max": int(counts.max())})
        previous_users, previous_pairs = users, pairs
    if protected_sha256 is not None:
        if root is None:
            raise DataValidationError("Root required for protected artifact verification")
        for name, expected in protected_sha256.items():
            if file_sha256(root / name) != expected:
                raise DataValidationError(f"Protected artifact changed: {name}")
    comparison = scaling_summary(datasets)
    comparison["clean_interactions"] = [data.report["input"]["rows"] for data in datasets.values()]
    comparison["positive_interactions"] = [data.report["input"]["positive_rows"] for data in datasets.values()]
    comparison["train_csr_shape"] = [str(data.matrix.shape) for data in datasets.values()]
    return {"comparison": comparison, "coverage": pd.DataFrame(coverage), "matrices": pd.DataFrame(matrices),
        "train_activity": pd.DataFrame(activity),
        "checks": {"nested_train": True, "fixed_evaluation": True, "user_disjoint": True,
                   "train_only_vocabulary": True, "warm_only_context_csr": True,
                   "removal_accounting": True, "protected_artifacts_unchanged": protected_sha256 is not None}}


def load_scaling_resources(directory):
    """Read aggregate telemetry only; a missing monitor result is explicit."""
    path = Path(directory) / "controlled_run_resources.json"
    if not path.exists():
        return {"status": "not_recorded"}
    value = json.loads(path.read_text(encoding="utf-8"))
    result = {key: value[key] for key in ("exit_code", "elapsed_seconds", "peak_working_set_bytes", "automatic_retries",
              "memory_measurement_note", "sampling_observed_peak_working_set_bytes")
              if key in value}
    verification = Path(directory) / "local_verification.json"
    if verification.exists():
        saved = json.loads(verification.read_text(encoding="utf-8"))
        result.update({key: saved[key] for key in ("full_passes_this_task", "verified_payload_bytes", "cache_bytes",
                       "sample_artifact_bytes", "cache_reuse") if key in saved})
    return result


def save_scaling_audit(directory, datasets, audit):
    """Persist only public aggregates and dataset fingerprints, never user histories."""
    report = {key: frame.to_dict(orient="records") for key, frame in audit.items() if isinstance(frame, pd.DataFrame)}
    report["checks"] = audit["checks"]
    report["dataset_fingerprints"] = {name: data.report["scaling_fingerprint"] for name, data in datasets.items()}
    report["resources"] = load_scaling_resources(directory)
    atomic_json(Path(directory) / "aggregate_audit.json", report)
    return report
