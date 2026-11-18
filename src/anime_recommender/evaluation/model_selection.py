"""Paired user bootstrap and exposure analysis of saved validation reports only."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from anime_recommender.evaluation.ranking_evaluator import model_fingerprint
from anime_recommender.evaluation.reports import json_safe, load_report
from anime_recommender.inference.model_versions import load_version_registry
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.training_data import DataValidationError
from anime_recommender.training.artifacts import load_models

VERSIONS = ("1k", "5k", "10k")
PAIRS = (("5k", "1k"), ("10k", "1k"), ("10k", "5k"))
METRICS = ("recall", "ndcg")


def paired_bootstrap(reference, candidate, *, resamples=10000, seed=42, tie_tolerance=1e-12):
    """Align by user ID; resample paired user differences, never independent models.

    Undefined values must have identical masks; they are excluded, not set to zero.
    Relative delta = difference of macro means / reference macro mean.
    """
    if type(resamples) is not int or resamples < 1 or type(seed) is not int:
        raise DataValidationError("Positive integer resamples and integer seed required")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise DataValidationError("Invalid tie tolerance")
    if reference.index.has_duplicates or candidate.index.has_duplicates or set(reference.index) != set(candidate.index):
        raise DataValidationError("Paired metrics require identical unique user IDs")
    reference = reference.sort_index().astype(float)
    candidate = candidate.reindex(reference.index).astype(float)
    if not reference.isna().equals(candidate.isna()):
        raise DataValidationError("Undefined metric masks differ for fixed targets")
    keep = reference.notna()
    left, right = reference.loc[keep].to_numpy(), candidate.loc[keep].to_numpy()
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise DataValidationError("Nonfinite user metrics")
    if ((left < 0) | (left > 1) | (right < 0) | (right > 1)).any():
        raise DataValidationError("User metrics must be in [0, 1]")
    n = len(left)
    result = {"eligible_users": n, "undefined_users": len(reference)-n,
        "resamples": resamples, "seed": seed, "tie_tolerance": tie_tolerance}
    if not n:
        return {**result, **dict.fromkeys(("reference_mean", "candidate_mean", "absolute_delta", "relative_delta",
            "ci95_low", "ci95_high", "improved_share", "worsened_share", "unchanged_share"))}
    difference = right - left
    # Reset RNG for every comparison: identical eligible ID sets get the same draws
    # across every pair AND metric. The bounded block avoids a B x N allocation.
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    for offset in range(0, resamples, 256):
        size = min(256, resamples-offset)
        draws = rng.integers(0, n, size=(size, n))
        boot[offset:offset+size] = difference[draws].mean(axis=1)
    delta, base = float(difference.mean()), float(left.mean())
    return {**result, "reference_mean": base, "candidate_mean": float(right.mean()),
        "absolute_delta": delta, "relative_delta": delta/base if base else None,
        "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975)),
        "improved_share": float((difference > tie_tolerance).mean()),
        "worsened_share": float((difference < -tie_tolerance).mean()),
        "unchanged_share": float((np.abs(difference) <= tie_tolerance).mean())}


def popularity_exposure(recommendations, popularity, items, train_users, k=10):
    """Occurrence-weighted popularity with train-user-rate and rank normalization.

    Popularity descending, dataset ID ascending breaks ties. Percentile rank is
    100*(N-zero_based_position)/N; exact top ceil(10%*N) items have percentile >90.
    """
    if type(train_users) is not int or train_users < 1 or type(k) is not int or k < 1:
        raise DataValidationError("Positive train user count and K required")
    voc = set(items.dataset_anime_id)
    if len(voc) != len(items) or not voc:
        raise DataValidationError("Nonempty unique item vocabulary required")
    pop = popularity[["dataset_anime_id", "unique_users"]].copy()
    if pop.dataset_anime_id.duplicated().any() or set(pop.dataset_anime_id) != voc:
        raise DataValidationError("Popularity/vocabulary mismatch")
    if not np.isfinite(pop.unique_users).all() or not pop.unique_users.between(1, train_users).all():
        raise DataValidationError("Invalid unique train-user popularity counts")
    pop = pop.sort_values(["unique_users", "dataset_anime_id"], ascending=[False, True]).reset_index(drop=True)
    n = len(pop)
    pop["popularity_percentile"] = 100*(n-np.arange(n))/n
    pop["train_user_share"] = pop.unique_users/train_users
    pop["top_decile"] = np.arange(n) < math.ceil(n*.1)
    recs = recommendations.loc[recommendations["rank"].le(k)].copy()
    if recs.duplicated(["dataset_user_id", "dataset_anime_id"]).any() or not set(recs.dataset_anime_id) <= voc:
        raise DataValidationError("Duplicate or out-of-vocabulary recommendation")
    if recs.duplicated(["dataset_user_id", "rank"]).any() or not recs["rank"].ge(1).all():
        raise DataValidationError("Invalid recommendation ranks")
    exposed = recs.merge(pop, on="dataset_anime_id", how="left", validate="many_to_one")
    deciles = pd.cut(exposed.popularity_percentile, bins=np.arange(0, 101, 10),
                     labels=[f"{i}-{i+10}" for i in range(0, 100, 10)], right=True)
    counts = deciles.value_counts(sort=False)
    total = len(exposed)
    distribution = pd.DataFrame({"percentile_bin": counts.index.astype(str), "recommendations": counts.to_numpy()})
    distribution["share"] = distribution.recommendations/total if total else None
    cutoff = pop.iloc[math.ceil(n*.1)-1].unique_users
    return {"catalog_items": n, "recommendation_occurrences": total,
        "unique_recommended_items": int(recs.dataset_anime_id.nunique()),
        "catalog_coverage_10": recs.dataset_anime_id.nunique()/n,
        "mean_unique_train_users": float(exposed.unique_users.mean()) if total else None,
        "mean_train_user_share": float(exposed.train_user_share.mean()) if total else None,
        "mean_popularity_percentile": float(exposed.popularity_percentile.mean()) if total else None,
        "top10_item_count": math.ceil(n*.1), "top10_recommendation_share": float(exposed.top_decile.mean()) if total else None,
        "top10_boundary_popularity": int(cutoff), "items_tied_at_boundary": int(pop.unique_users.eq(cutoff).sum())}, distribution


def aligned_full_metrics(reports):
    """Validate identity, undefined masks and macro consistency of stored results."""
    aligned, identities = {}, set()
    expected_ids, expected_targets = None, None
    for version in VERSIONS:
        report = reports[version]
        metadata = report.metadata
        if metadata.get("split") != "validation" or metadata.get("franchise_filtering") is not False:
            raise DataValidationError("Require validation-only reports without franchise filtering")
        identities.add(metadata.get("fixed_validation_fingerprint"))
        table = report.tables["user_metrics"]
        selected = table.loc[table.model_name.eq("als") & table["mode"].eq("full") & table.k.eq(10)]
        if selected.dataset_user_id.duplicated().any():
            raise DataValidationError("Duplicate per-user ALS metrics")
        users = report.tables["users"].set_index("dataset_user_id").sort_index()
        ids = set(users.index)
        if users.index.has_duplicates or set(selected.dataset_user_id) != ids:
            raise DataValidationError("Per-user metrics do not cover the fixed cohort")
        if expected_ids is not None and ids != expected_ids:
            raise DataValidationError("Validation user IDs differ across reports")
        targets = users.positive_targets
        if expected_targets is not None and not targets.equals(expected_targets):
            raise DataValidationError("Fixed positive target counts changed")
        expected_ids, expected_targets = ids, targets
        selected = selected.set_index("dataset_user_id").sort_index()[list(METRICS)]
        expected_undefined = targets.eq(0)
        for metric in METRICS:
            if not selected[metric].isna().equals(expected_undefined):
                raise DataValidationError("Undefined metric does not match empty target set")
            macro = report.tables["metrics"]
            macro = macro.loc[macro.model_name.eq("als") & macro["mode"].eq("full") & macro.k.eq(10)]
            if len(macro) != 1 or not np.isclose(selected[metric].mean(), macro[metric].iloc[0], atol=1e-12, rtol=0, equal_nan=True):
                raise DataValidationError("Stored macro and per-user metrics disagree")
        aligned[version] = selected
    if len(identities) != 1 or None in identities:
        raise DataValidationError("Missing/different fixed validation fingerprint")
    return aligned


def run_selection_audit(root, *, resamples=10000, seed=42):
    root = Path(root).resolve()
    registry = load_version_registry(root)
    base = (root / registry["comparison_dir"]).resolve()
    if not base.is_relative_to(root / "artifacts/evaluation"):
        raise DataValidationError("Unsafe evaluation directory")
    reports = {version: load_report(base/version) for version in VERSIONS}
    if len({r.metadata["protocol_version"] for r in reports.values()}) != 1:
        raise DataValidationError("Evaluation protocols differ")
    aligned = aligned_full_metrics(reports)
    bootstrap = []
    for candidate, reference in PAIRS:
        for metric in METRICS:
            bootstrap.append({"candidate": candidate, "reference": reference, "metric": metric,
                **paired_bootstrap(aligned[reference][metric], aligned[candidate][metric], resamples=resamples, seed=seed)})
    diversity, distributions, sources = [], [], {}
    for version in VERSIONS:
        path = root/registry["versions"][version]["model_dir"]
        bundle = load_models(path)
        report = reports[version]
        if (model_fingerprint(bundle) != report.metadata["model_fingerprint"] or
                bundle.metadata["train_fingerprint"] != report.metadata["train_fingerprint"]):
            raise DataValidationError("Saved model and evaluation fingerprints differ")
        recs = report.tables["recommendations"].loc[lambda f: f.model_name.eq("als")]
        stats, distribution = popularity_exposure(recs, bundle.popularity, bundle.items, len(bundle.train_users))
        saved = report.tables["coverage"].loc[lambda f: f.model_name.eq("als") & f.k.eq(10)]
        if len(saved) != 1 or not np.isclose(saved.catalog_coverage.iloc[0], stats["catalog_coverage_10"], atol=1e-12):
            raise DataValidationError("Saved and audited catalog coverage differ")
        if not np.isclose(saved.mean_train_popularity.iloc[0], stats["mean_unique_train_users"], atol=1e-12):
            raise DataValidationError("Saved and audited popularity means differ")
        diversity.append({"model_version": version, "train_users": len(bundle.train_users), **stats})
        distributions.append(distribution.assign(model_version=version))
        sources[version] = {"report_sha256": file_sha256(base/version/"report.json"),
            "model_metadata_sha256": file_sha256(path/"metadata.json"),
            "model_fingerprint": report.metadata["model_fingerprint"],
            "dataset_fingerprint": report.metadata["dataset_fingerprint"],
            "fixed_validation_fingerprint": report.metadata["fixed_validation_fingerprint"]}
    tables = {"bootstrap": pd.DataFrame(bootstrap), "diversity": pd.DataFrame(diversity),
              "popularity_distribution": pd.concat(distributions, ignore_index=True)}
    output = base/"final_selection"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        atomic_parquet(table, output/(name+".parquet"))
    metadata = {"split": "validation", "test_used": False, "models_retrained": False,
        "resamples": resamples, "seed": seed, "confidence_interval": "paired user percentile bootstrap, 2.5/97.5%; conditional on fixed fitted models",
        "multiple_comparisons": "six pointwise intervals; no simultaneous-coverage or multiplicity correction",
        "ties": "abs(per-user metric difference)<=1e-12 for classification only; raw differences in bootstrap",
        "popularity": "occurrence weighted; unique positive train-user counts divided by retained train users; full-vocabulary rank percentiles",
        "top10_policy": "exact ceil(10%*vocabulary), sort popularity descending / dataset ID ascending for ties",
        "configured_default": registry["default_version"], "sources": sources,
        "tables": {name: frame.to_dict("records") for name, frame in tables.items()},
        "files_sha256": {name+".parquet": file_sha256(output/(name+".parquet")) for name in tables}}
    atomic_json(output/"report.json", json_safe(metadata))
    return tables, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tables, _ = run_selection_audit(Path.cwd(), resamples=args.resamples, seed=args.seed)
    for name, frame in tables.items():
        print(name)
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
