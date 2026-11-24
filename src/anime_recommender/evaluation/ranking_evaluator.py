"""Fixed-model validation evaluation. No training, API calls or test file reads."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anime_recommender.evaluation.cohorts import COHORTS, assign_cohorts, history_distribution
from anime_recommender.evaluation.coverage import catalog_coverage, context_target_coverage
from anime_recommender.evaluation.ranking_metrics import METRICS, ranking_metrics
from anime_recommender.evaluation.recommendation_checks import generate_validation_recommendations
from anime_recommender.evaluation.reports import EvaluationReport, load_report, save_report
from anime_recommender.artifact_files import file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError, positive_ids, require_columns
from anime_recommender.artifact_io import load_models, mapping_fingerprint
from anime_recommender.validation_inputs import load_validation_view


PROTOCOL_VERSION = "ARE-007A-v1"
PROTOCOL = {"split": "validation only; user-disjoint; raw context/target split before positives",
    "candidates": "train vocabulary minus ALL observed context (including low/zero ratings)",
    "generation": "one max-K list per model/user; same prefixes for all K and both modes",
    "full": "all positive targets, including cold items",
    "warm": "warm targets only; no context-based eligibility condition",
    "undefined": "all five ranking metrics undefined for empty relevant target set",
    "precision": "hits/K, including when fewer than K candidates are available",
    "ap": "sum(precision at each hit rank)/min(K, relevant target count)",
    "ndcg": "binary DCG=sum(1/log2(rank+1)); ideal has min(K, relevant count) hits",
    "macro": "exclude undefined values; publish metric-specific eligible user counts",
    "coverage": "distinct recommended items/train vocabulary; popularity mean over displayed occurrences",
    "cohorts": "distinct warm positive context only; empty group metrics are null",
    "cohort_target_sizes": "Full and warm target mean/median over ALL cohort users, including zero-target users; descriptive only",
    "fallback": "als and als_fallback reported separately; fallback only without warm context"}


def validate_ks(ks):
    if not isinstance(ks, (list, tuple)) or not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("K values must be a nonempty list of positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("Duplicate K values")
    return sorted(ks)


def _validate_inputs(bundle, inputs):
    if mapping_fingerprint(bundle.items) != mapping_fingerprint(inputs.items):
        raise DataValidationError("Evaluation/model vocabulary mismatch")
    if bundle.metadata["train_fingerprint"] != inputs.train_fingerprint:
        raise DataValidationError("Evaluation/model training fingerprint mismatch")
    ids = positive_ids(inputs.validation_users.dataset_user_id, "validation users")
    if ids.empty or ids.duplicated().any():
        raise DataValidationError("Require unique nonempty validation users")
    users = set(ids)
    if users & set(inputs.train_users.dataset_user_id):
        raise DataValidationError("Train-validation user leakage")
    pairs = []
    for frame in (inputs.validation_context, inputs.observed_context, inputs.validation_target):
        require_columns(frame, ["dataset_user_id", "dataset_anime_id"], "evaluation table")
        for col in ("dataset_user_id", "dataset_anime_id"):
            positive_ids(frame[col], col)
        if not set(frame.dataset_user_id) <= users or frame.duplicated(["dataset_user_id", "dataset_anime_id"]).any():
            raise DataValidationError("Unknown evaluation user or duplicate interaction")
        pairs.append(set(zip(frame.dataset_user_id, frame.dataset_anime_id)))
    if not pairs[0] <= pairs[1] or pairs[1] & pairs[2]:
        raise DataValidationError("Invalid context/observed/target separation")


def evaluation_fingerprint(inputs):
    digest = hashlib.sha256(inputs.train_fingerprint.encode())
    for frame in (inputs.validation_users, inputs.validation_context, inputs.observed_context, inputs.validation_target):
        sorted_frame = frame.sort_values(list(frame.columns)).reset_index(drop=True)
        digest.update(sorted_frame.to_json(orient="split", double_precision=15).encode())
    return digest.hexdigest()


def model_fingerprint(bundle):
    """Content identity even for an in-memory bundle with no saved file hashes."""
    payload = {"mapping": mapping_fingerprint(bundle.items),
        "train_fingerprint": bundle.metadata["train_fingerprint"],
        "als_config": bundle.metadata["training_config"]["als"],
        "popularity": bundle.popularity.sort_values("dataset_anime_id")[
            ["dataset_anime_id", "unique_users"]].to_dict("list")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for factors in (bundle.als.item_factors, bundle.als.user_factors):
        digest.update(str((factors.shape, str(factors.dtype))).encode())
        digest.update(np.ascontiguousarray(factors).tobytes())
    return digest.hexdigest()


def _add_fallback(recs, statuses):
    unavailable = set(statuses.loc[statuses.model_name.eq("als") & ~statuses.available, "dataset_user_id"])
    chosen = recs.loc[(recs.model_name.eq("als") & ~recs.dataset_user_id.isin(unavailable)) |
                      (recs.model_name.eq("popularity") & recs.dataset_user_id.isin(unavailable))].copy()
    chosen["model_name"] = "als_fallback"
    status = statuses.loc[statuses.model_name.eq("als")].copy()
    pop_counts = statuses.loc[statuses.model_name.eq("popularity")].set_index("dataset_user_id").recommendations
    missing = status.dataset_user_id.isin(unavailable)
    status.loc[missing, "recommendations"] = status.loc[missing, "dataset_user_id"].map(pop_counts)
    status["fallback_used"] = missing
    status["selected_model"] = np.where(missing, "popularity", "als")
    status["available"] = True
    status["model_name"] = "als_fallback"
    pieces = [frame for frame in (recs, chosen) if not frame.empty]
    return (pd.concat(pieces, ignore_index=True) if pieces else recs), pd.concat([statuses, status], ignore_index=True)


def _aggregate(group, users, name, mode, k):
    row = {"model_name": name, "mode": mode, "k": k, **context_target_coverage(users)}
    for metric in METRICS:
        eligible = int(group[metric].notna().sum())
        row[metric] = float(group[metric].mean()) if eligible else None
        row[metric + "_eligible_users"] = eligible
    return row


def _examples(recs, per_user, targets, k):
    values = per_user.loc[per_user.model_name.eq("als") & per_user["mode"].eq("full") & per_user.k.eq(k)]
    selected, frames = [], []
    target_sets = {int(uid): set(g.dataset_anime_id) for uid, g in targets.groupby("dataset_user_id")}
    for label, condition in (("success", values.recall.gt(0)), ("failure", values.recall.eq(0))):
        candidates = values.loc[condition].sort_values("dataset_user_id")
        if candidates.empty:
            continue
        uid = int(candidates.dataset_user_id.iloc[0])
        selected.append({"case": label, "dataset_user_id": uid, "model_name": "als", "k": k})
        frame = recs.loc[recs.model_name.eq("als") & recs.dataset_user_id.eq(uid) & recs["rank"].le(k)].copy()
        frame["case"] = label
        frame["is_hit"] = frame.dataset_anime_id.isin(target_sets.get(uid, set()))
        frames.append(frame)
    example_users = pd.DataFrame(selected, columns=["case", "dataset_user_id", "model_name", "k"])
    examples = pd.concat(frames, ignore_index=True) if frames else recs.iloc[:0].assign(case=pd.Series(dtype=str), is_hit=pd.Series(dtype=bool))
    return examples, example_users


def evaluate_models(bundle, inputs, ks=(5, 10, 15, 20), include_als_fallback=True):
    ks = validate_ks(ks)
    if type(include_als_fallback) is not bool:
        raise ValueError("include_als_fallback must be boolean")
    _validate_inputs(bundle, inputs)
    users = assign_cohorts(inputs.validation_users, inputs.validation_context, bundle.items)
    before_items, before_users = bundle.als.item_factors.copy(), bundle.als.user_factors.copy()
    # No target argument. Recommendations are generated once and then frozen.
    recs, statuses = generate_validation_recommendations(bundle.recommenders(), inputs.validation_users,
        inputs.validation_context, inputs.observed_context, bundle.items, max(ks))
    if not np.array_equal(before_items, bundle.als.item_factors) or not np.array_equal(before_users, bundle.als.user_factors):
        raise DataValidationError("Evaluation changed trained factors")
    statuses = statuses.rename(columns={"als_available": "available"})
    statuses["fallback_used"] = False
    statuses["selected_model"] = statuses.model_name
    if include_als_fallback:
        recs, statuses = _add_fallback(recs, statuses)
    vocabulary = set(bundle.items.dataset_anime_id)
    targets = {int(uid): set(frame.dataset_anime_id) for uid, frame in inputs.validation_target.groupby("dataset_user_id")}
    users["positive_targets"] = users.dataset_user_id.map(lambda uid: len(targets.get(uid, set())))
    users["cold_targets"] = users.dataset_user_id.map(lambda uid: len(targets.get(uid, set()) - vocabulary))
    selected_lists = {(int(uid), name): g.sort_values("rank").dataset_anime_id.tolist()
                      for (uid, name), g in recs.groupby(["dataset_user_id", "model_name"])}
    rows = []
    for status in statuses.to_dict("records"):
        uid, name = status["dataset_user_id"], status["model_name"]
        relevant = targets.get(uid, set())
        ordered = selected_lists.get((uid, name), [])
        for k in ks:
            for mode, wanted in (("full", relevant), ("warm", relevant & vocabulary)):
                rows.append({"dataset_user_id": uid, "model_name": name, "mode": mode, "k": k,
                    **ranking_metrics(ordered, wanted, k)})
    per_user = pd.DataFrame(rows).merge(users, on="dataset_user_id", validate="many_to_one")
    metrics, cohort_rows, coverages = [], [], []
    popularity = bundle.popularity.set_index("dataset_anime_id").unique_users
    for (name, mode, k), group in per_user.groupby(["model_name", "mode", "k"], sort=True):
        metrics.append(_aggregate(group, users, name, mode, int(k)))
        for cohort in COHORTS:
            cohort_users = users.loc[users.cohort.eq(cohort)]
            cohort_rows.append({**_aggregate(group.loc[group.cohort.eq(cohort)], cohort_users, name, mode, int(k)), "cohort": cohort})
        if mode == "full":
            exposed = recs.loc[recs.model_name.eq(name) & recs["rank"].le(k)]
            coverages.append({"model_name": name, "k": int(k), **catalog_coverage(exposed.dataset_anime_id, vocabulary),
                "mean_train_popularity": float(exposed.dataset_anime_id.map(popularity).mean()) if len(exposed) else None})
    examples, example_users = _examples(recs, per_user, inputs.validation_target, 10 if 10 in ks else ks[0])
    metadata = {"protocol_version": PROTOCOL_VERSION, "protocol": PROTOCOL,
        "split": "validation", "ks": ks, "include_als_fallback": include_als_fallback,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_fingerprint": inputs.train_fingerprint, "dataset_fingerprint": evaluation_fingerprint(inputs),
        "model_fingerprint": model_fingerprint(bundle),
        "model_configuration": bundle.metadata["training_config"], "dataset_configuration": inputs.dataset_config,
        "model_files_sha256": bundle.metadata.get("files_sha256", {}),
        "versions": {name: version(name) for name in ("implicit", "numpy", "pandas", "scipy", "pyarrow")},
        "systems": {"popularity": "train-only popularity", "als": "ALS-only; no fallback",
                    **({"als_fallback": "ALS with explicit popularity fallback"} if include_als_fallback else {})},
        "factors_unchanged": True, "test_evaluated": False}
    return EvaluationReport(metadata, {"metrics": pd.DataFrame(metrics), "cohorts": pd.DataFrame(cohort_rows),
        "user_metrics": per_user, "users": users, "history_distribution": history_distribution(users),
        "coverage": pd.DataFrame(coverages), "recommendations": recs, "statuses": statuses,
        "examples": examples, "example_users": example_users})


def run_evaluation(root, config_path="configs/evaluation.json", persist=True):
    root = Path(root).resolve()
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    if set(config) != {"dataset_dir", "model_dir", "catalog_path", "output_dir", "split", "ks", "include_als_fallback"} or config["split"] != "validation":
        raise ValueError("Only the configured validation protocol is supported")
    for key, parent in (("dataset_dir", "data"), ("catalog_path", "data"), ("model_dir", "artifacts"), ("output_dir", "artifacts/evaluation")):
        path = Path(config[key])
        if path.is_absolute() or not (root / path).resolve().is_relative_to((root / parent).resolve()):
            raise ValueError("Evaluation paths must stay in their designated local directories")
    validate_ks(config["ks"])
    bundle = load_models(root / config["model_dir"])
    inputs = load_validation_view(root / config["dataset_dir"], bundle)
    report = evaluate_models(bundle, inputs, config["ks"], config["include_als_fallback"])
    catalog_path = root / config["catalog_path"]
    if file_sha256(catalog_path) != inputs.catalog_sha256:
        raise DataValidationError("Example catalog differs from certified dataset")
    catalog = pd.read_parquet(catalog_path)[["dataset_anime_id", "title"]]
    report.tables["examples"] = report.tables["examples"].merge(catalog, on="dataset_anime_id", how="left", validate="many_to_one")
    report.metadata["validation_files_sha256"] = {name: file_sha256(root / config["dataset_dir"] / name) for name in (
        "validation_context.parquet", "validation_target.parquet", "validation_observed_context.parquet", "validation_user_mapping.parquet")}
    report.metadata["evaluation_config"] = config
    if persist:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = root / config["output_dir"] / (stamp + "-" + report.metadata["dataset_fingerprint"][:8])
        report.metadata["report_dir"] = str(directory.relative_to(root))
        save_report(report, directory)
        loaded = load_report(directory)
        for name in report.tables:
            pd.testing.assert_frame_equal(report.tables[name], loaded.tables[name])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.json")
    args = parser.parse_args()
    report = run_evaluation(Path(__file__).resolve().parents[3], args.config)
    print(report.tables["metrics"].to_string(index=False))
    print("Saved:", report.metadata["report_dir"])


if __name__ == "__main__":
    main()
