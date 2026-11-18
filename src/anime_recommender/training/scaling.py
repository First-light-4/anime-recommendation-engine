"""Fixed evaluation, growing train. Local preparation only; never fits a model."""

import argparse
import hashlib
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.sparse import save_npz

from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids
from anime_recommender.training.dataset import PreparedDataset, load_dataset, prepare_dataset

PROTOCOL = "fixed-evaluation-bottom-k-v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def nested_users(user_ids, baseline_ids, size, seed):
    """Keep baseline explicitly, then rank new IDs by the ARE-004 hash priority."""
    available = set(positive_ids(pd.Series(list(user_ids), dtype="object"), "sample users"))
    baseline = set(positive_ids(pd.Series(list(baseline_ids), dtype="object"), "baseline users"))
    if type(size) is not int or size < len(baseline) or size > len(available) or not baseline <= available:
        raise DataValidationError("Requested size unavailable or baseline users missing")
    ordered = sorted(available - baseline, key=lambda uid: (
        hashlib.sha256(f"{seed}:{uid}".encode()).digest(), uid))
    return sorted(baseline | set(ordered[:size - len(baseline)]))


def _same(left, right, label):
    try:
        assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    except AssertionError as error:
        raise DataValidationError(f"Fixed baseline/evaluation changed: {label}") from error


def prepare_scaling(interactions, mapping, baseline, selected_users, config, *, previous=None):
    """Input is already cleaned complete histories; targets never enter train filtering."""
    original = baseline.tables["user_assignments"][["dataset_user_id", "split"]]
    selected = set(selected_users)
    if not set(original.dataset_user_id) <= selected:
        raise DataValidationError("Baseline users missing from selected population")
    if config["positive_threshold"] != 5 or config["min_user_positives"] != 5 or config["min_item_users"] != 5:
        raise DataValidationError("Scaling requires unchanged threshold=5 and train minima=5/5")
    if baseline.report.get("dataset_version") != 2:
        raise DataValidationError("Incompatible baseline dataset version")
    for key in ("positive_threshold", "min_user_positives", "min_item_users", "seed", "evaluation_protocol"):
        if config[key] != baseline.report["config"][key]:
            raise DataValidationError(f"Incompatible baseline configuration: {key}")
    assignments = pd.concat([original, pd.DataFrame({
        "dataset_user_id": sorted(selected - set(original.dataset_user_id)), "split": "train"})], ignore_index=True)
    selected_rows = interactions.loc[interactions.dataset_user_id.isin(selected)].copy()
    holdouts = {name: tuple(baseline.tables[f"{name}_observed_{part}"] for part in ("context", "target"))
                for name in ("validation", "test")}
    for name, parts in holdouts.items():
        expected = pd.concat(parts).sort_values(["dataset_user_id", "dataset_anime_id"]).reset_index(drop=True)
        ids = set(original.loc[original.split.eq(name), "dataset_user_id"])
        actual = selected_rows.loc[selected_rows.dataset_user_id.isin(ids)].sort_values(
            ["dataset_user_id", "dataset_anime_id"]).reset_index(drop=True)
        _same(actual, expected, name + " raw histories")
    result = prepare_dataset(selected_rows, mapping, config,
                             fixed_assignments=assignments, fixed_holdouts=holdouts)
    for name in ("validation", "test"):
        for part in ("context", "target", "observed_context", "observed_target"):
            key = f"{name}_{part}"
            _same(result.tables[key].drop(columns="is_cold_item", errors="ignore"),
                  baseline.tables[key].drop(columns="is_cold_item", errors="ignore"), key)
    for smaller in (baseline, previous):
        if smaller is None:
            continue
        if not set(smaller.tables["train_user_mapping"].dataset_user_id) <= set(result.tables["train_user_mapping"].dataset_user_id):
            raise DataValidationError("Retained train users are not nested")
        old = smaller.tables["train_interactions"]
        new = result.tables["train_interactions"]
        joined = old.merge(new, on=["dataset_user_id", "dataset_anime_id"], suffixes=("_old", "_new"))
        if len(joined) != len(old) or not joined.rating_old.eq(joined.rating_new).all():
            raise DataValidationError("Retained train interactions are not nested")
    result.report["checks"].update(fixed_evaluation=True, nested_train=True, threshold_unchanged=True)
    return result


def save_scaling_dataset(directory, prepared, manifest, *, baseline_directory=None):
    """Write immutable experiment. Completion report last; repeated runs verify, not overwrite."""
    directory = Path(directory)
    identity = fingerprint(manifest)
    if directory.exists():
        loaded = load_scaling_dataset(directory)
        if loaded.report["scaling_fingerprint"] != identity:
            raise DataValidationError("Experiment already exists with another fingerprint")
        return loaded
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scaling-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        references = {}
        for name, frame in prepared.tables.items():
            filename = name + ".parquet"
            shared = baseline_directory is not None and name in {
                f"{cohort}_observed_{part}" for cohort in ("validation", "test") for part in ("context", "target")}
            if shared:
                path = Path(baseline_directory).resolve() / filename
                references[filename] = {"path": str(path), "sha256": file_sha256(path)}
            else:
                atomic_parquet(frame, staging / filename)
        save_npz(staging / "train_matrix.npz", prepared.matrix)
        for name, matrix in prepared.context_matrices.items():
            save_npz(staging / f"{name}_context_matrix.npz", matrix)
        atomic_json(staging / "sampling_manifest.json", manifest)
        report = dict(prepared.report)
        report.update(scaling_protocol=PROTOCOL, scaling_fingerprint=identity, shared_artifacts=references)
        report["artifact_sha256"] = {path.name: file_sha256(path) for path in sorted(staging.iterdir()) if path.is_file()}
        atomic_json(staging / "dataset_report.json", report)
        staging.replace(directory)
    return load_scaling_dataset(directory)


def load_scaling_dataset(directory):
    """Checksum-verified common interface for baseline and expanded experiments."""
    directory = Path(directory)
    report = json.loads((directory / "dataset_report.json").read_text(encoding="utf-8"))
    if report.get("scaling_protocol") != PROTOCOL or report.get("dataset_version") != 2 or report.get("status") != "success":
        raise DataValidationError("Incompatible or incomplete scaling dataset")
    if "baseline_reference" in report:
        reference = report["baseline_reference"]
        if file_sha256(Path(reference) / "dataset_report.json") != report["baseline_report_sha256"]:
            raise DataValidationError("Baseline fingerprint mismatch")
        data = load_dataset(reference)
        return PreparedDataset(data.tables, data.matrix, report, data.context_matrices)
    # Standard ARE-005 loader verifies locally stored artifacts and matrices.
    data = load_dataset(directory)
    for filename, reference in report.get("shared_artifacts", {}).items():
        path = Path(reference["path"])
        if file_sha256(path) != reference["sha256"]:
            raise DataValidationError("Shared evaluation fingerprint mismatch")
        data.tables[Path(filename).stem] = pd.read_parquet(path)
    manifest = json.loads((directory / "sampling_manifest.json").read_text(encoding="utf-8"))
    if fingerprint(manifest) != report["scaling_fingerprint"]:
        raise DataValidationError("Sampling manifest fingerprint mismatch")
    return data


def register_baseline(baseline_directory, output):
    """Reference existing immutable files rather than duplicate the baseline."""
    baseline_directory, output = Path(baseline_directory).resolve(), Path(output)
    baseline = load_dataset(baseline_directory)
    report = dict(baseline.report)
    report["sampling_manifest"] = {"protocol": PROTOCOL, "strategy": "preserve-certified-baseline",
        "seed": baseline.report["config"]["seed"], "sample_size": len(baseline.tables["user_assignments"]),
        "selected_user_ids": sorted(map(int, baseline.tables["user_assignments"].dataset_user_id)),
        "source_sha256": baseline.report.get("source_sha256", {})}
    report.update(scaling_protocol=PROTOCOL, baseline_reference=str(baseline_directory),
                  baseline_report_sha256=file_sha256(baseline_directory / "dataset_report.json"))
    report["scaling_fingerprint"] = fingerprint({"baseline_report": report["baseline_report_sha256"], "protocol": PROTOCOL})
    if output.exists():
        existing = load_scaling_dataset(output)
        if existing.report["scaling_fingerprint"] != report["scaling_fingerprint"]:
            raise DataValidationError("Baseline registration differs")
        return existing
    output.mkdir(parents=True)
    atomic_json(output / "dataset_report.json", report)
    return load_scaling_dataset(output)


def scaling_summary(datasets):
    rows = []
    for name, dataset in datasets.items():
        rows.append({"sample": name, "selected_users": len(dataset.tables["user_assignments"]),
            "train_users": dataset.matrix.shape[0], "train_items": dataset.matrix.shape[1],
            "train_interactions": dataset.matrix.nnz, "sparsity": dataset.report["matrix"]["sparsity"],
            "validation_cold_target_share": dataset.report["evaluation"]["validation"]["target"]["cold_share"],
            "test_cold_target_share": dataset.report["evaluation"]["test"]["target"]["cold_share"]})
    return pd.DataFrame(rows)


def synthetic_scaling_demo():
    """Small, explicitly synthetic offline example for Notebook 07 and smoke checks."""
    config = {"seed": 42, "positive_threshold": 5, "feedback": "binary", "confidence": 1.,
        "min_user_positives": 5, "min_item_users": 5, "train_fraction": .8,
        "validation_fraction": .1, "test_fraction": .1, "context_fraction": .8,
        "min_context": 1, "min_target": 1, "evaluation_protocol": "raw_holdout_then_positive_v2"}
    rows = pd.DataFrame([(user, item, 8.) for user in range(1, 31) for item in range(1, 9)],
                        columns=["dataset_user_id", "dataset_anime_id", "rating"])
    mapping = pd.DataFrame({"dataset_anime_id": range(1, 9), "myanimelist_id": range(101, 109)})
    baseline = prepare_dataset(rows.loc[rows.dataset_user_id.le(10)], mapping, config)
    smaller_ids = nested_users(range(1, 31), range(1, 11), 20, 42)
    smaller = prepare_scaling(rows, mapping, baseline, smaller_ids, config)
    larger = prepare_scaling(rows, mapping, baseline, range(1, 31), config, previous=smaller)
    return {"synthetic_baseline": baseline, "synthetic_20": smaller, "synthetic_30": larger}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaling.json")
    parser.add_argument("--execute", action="store_true", help="Prepare from verified local caches only")
    parser.add_argument("--allow-full-scan", action="store_true", help="Explicit approval for one 10k source scan")
    args = parser.parse_args()
    from anime_recommender.ingestion.scaling import run_scaling
    print(json.dumps(run_scaling(Path.cwd(), args.config, execute=args.execute,
                                 allow_full_scan=args.allow_full_scan), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Scaling preparation failed: {error}") from None
