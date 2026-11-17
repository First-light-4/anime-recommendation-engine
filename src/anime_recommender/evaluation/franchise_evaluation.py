"""Separate partial-coverage validation diagnostic; never rewrites ARE-007A."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anime_recommender.evaluation.ranking_evaluator import evaluate_models
from anime_recommender.evaluation.recommendation_checks import diagnostic_metrics
from anime_recommender.evaluation.reports import json_safe
from anime_recommender.inference.franchise_filter import recommend_with_franchise
from anime_recommender.inference.recommendations import OUTPUT_COLUMNS
from anime_recommender.ingestion.training_data import atomic_json, atomic_parquet, file_sha256
from anime_recommender.preprocessing.franchise_mapping import load_franchise_resources
from anime_recommender.training.artifacts import load_models
from anime_recommender.training.inputs import load_training_inputs


def compare_franchise_filter(bundle, inputs, catalog, mapping, config, top_k=10):
    if not config["enabled"]:
        raise ValueError("Enable franchise filtering explicitly for this separate diagnostic")
    baseline = evaluate_models(bundle, inputs, ks=[top_k], include_als_fallback=False)
    base_recs = baseline.tables["recommendations"].loc[lambda f: f.model_name.eq("als")]
    base_status = baseline.tables["statuses"].loc[lambda f: f.model_name.eq("als")].rename(columns={"available": "als_available"})
    contexts = {int(uid): g.dataset_anime_id.tolist() for uid, g in inputs.validation_context.groupby("dataset_user_id")}
    observed = {int(uid): g.dataset_anime_id.tolist() for uid, g in inputs.observed_context.groupby("dataset_user_id")}
    mal_by_id = catalog.set_index("dataset_anime_id").myanimelist_id.to_dict()
    recommender = bundle.recommenders()["als"]
    before_items, before_users = bundle.als.item_factors.copy(), bundle.als.user_factors.copy()
    frames, status_rows, diagnostics = [], [], []
    for uid in inputs.validation_users.dataset_user_id:
        uid = int(uid)
        seen = observed.get(uid, [])
        seen_mal = [int(mal_by_id[item]) for item in seen if item in mal_by_id and pd.notna(mal_by_id[item])]
        # Ratings do not prove completion. Do not use targets or invent statuses.
        filtered = recommend_with_franchise(recommender, bundle.items, contexts.get(uid, []), mapping,
            seen, seen_mal, completed_mal_ids=[], top_k=top_k,
            max_per_franchise=config["max_recommendations_per_franchise"],
            candidate_multiplier=config["candidate_multiplier"])
        frame = filtered.recommendations.copy()
        frame.insert(0, "dataset_user_id", uid)
        if not frame.empty:
            frames.append(frame)
        row = base_status.loc[base_status.dataset_user_id.eq(uid)].iloc[0].to_dict()
        row["recommendations"] = len(frame)
        status_rows.append(row)
        original_ids = set(base_recs.loc[base_recs.dataset_user_id.eq(uid), "dataset_anime_id"])
        removed = len(original_ids - set(frame.dataset_anime_id))
        diagnostics.append({"dataset_user_id": uid, **filtered.diagnostics,
            "original_top_k_count": len(original_ids), "removed_from_original_top_k": removed,
            "removed_original_share": removed / len(original_ids) if original_ids else None})
    recs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dataset_user_id", *OUTPUT_COLUMNS])
    filtered_status = pd.DataFrame(status_rows)
    summaries = []
    for label, predictions, status in (("als_original", base_recs, base_status), ("als_franchise", recs, filtered_status)):
        summary, _ = diagnostic_metrics(predictions, status, inputs.validation_target,
            inputs.validation_users, bundle.items, top_k)
        summary["model_name"] = label
        summary["system"] = "ALS-only" if label == "als_original" else "ALS-only + partial franchise filter"
        summaries.append(summary)
    if not np.array_equal(before_items, bundle.als.item_factors) or not np.array_equal(before_users, bundle.als.user_factors):
        raise ValueError("Post-processing changed trained factors")
    diag = pd.DataFrame(diagnostics)
    covered = mapping.describe(bundle.items.myanimelist_id)
    original_count = int(diag.original_top_k_count.sum())
    report = {"protocol": "ARE-007C-yummy-order-v2" if mapping.snapshot.get("ordered_sequences") else "ARE-007C-partial-v1", "split": "validation", "top_k": top_k,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "relations_fingerprint": mapping.fingerprint,
        "dataset_fingerprint": baseline.metadata["dataset_fingerprint"],
        "model_fingerprint": baseline.metadata["model_fingerprint"],
        "completion_policy": "No completed proof in ratings-only validation; completed_mal_ids=[]; no inferred ratings-to-status conversion",
        "covered_train_items": int(covered.franchise_id.notna().sum()), "train_items": len(bundle.items),
        "franchise_coverage": float(covered.franchise_id.notna().mean()),
        "order_known_train_items": int(covered.order_known.sum()),
        "removed_original_top_k": int(diag.removed_from_original_top_k.sum()),
        "original_top_k_count": original_count,
        "removed_original_share": float(diag.removed_from_original_top_k.sum() / original_count) if original_count else None,
        "factors_unchanged": True, "test_evaluated": False,
        "source": mapping.snapshot.get("source"),
        "known_franchises": len(set(mapping.franchise.values())),
        "scope": "Partial targeted snapshot only; not full-catalog or completed-status online evaluation"}
    return {"summary": pd.concat(summaries, ignore_index=True), "diagnostics": diag,
            "original_recommendations": base_recs, "filtered_recommendations": recs,
            "mapping_coverage": covered, "metadata": report}


def run_franchise_evaluation(root):
    root = Path(root).resolve()
    settings = json.loads((root / "configs/franchise_evaluation.json").read_text(encoding="utf-8"))
    if set(settings) != {"base_config", "output_dir", "top_k"} or type(settings["top_k"]) is not int or settings["top_k"] < 1:
        raise ValueError("Invalid franchise diagnostic configuration")
    base_path = (root / settings["base_config"]).resolve()
    output_path = (root / settings["output_dir"]).resolve()
    if not base_path.is_relative_to(root / "configs") or not output_path.is_relative_to(root / "artifacts/evaluation/are007c"):
        raise ValueError("Keep franchise config/output separate from old evaluation artifacts")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["split"] != "validation":
        raise ValueError("Only validation is allowed")
    inputs = load_training_inputs(root / base["dataset_dir"])
    bundle = load_models(root / base["model_dir"], inputs.items, inputs.train_fingerprint)
    catalog_path = root / base["catalog_path"]
    if file_sha256(catalog_path) != inputs.catalog_sha256:
        raise ValueError("Franchise diagnostic catalog differs from certified source")
    catalog = pd.read_parquet(catalog_path)
    config, mapping = load_franchise_resources(root)
    result = compare_franchise_filter(bundle, inputs, catalog, mapping, config, settings["top_k"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_path / stamp
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if name != "metadata":
            atomic_parquet(frame, directory / (name + ".parquet"))
    result["metadata"]["metrics"] = result["summary"].to_dict("records")
    result["metadata"]["files_sha256"] = {name: file_sha256(directory / name) for name in
        [key + ".parquet" for key in result if key != "metadata"]}
    atomic_json(directory / "report.json", json_safe(result["metadata"]))
    return result


if __name__ == "__main__":
    result = run_franchise_evaluation(Path(__file__).resolve().parents[3])
    print(result["summary"].to_string(index=False))
    print(json.dumps(result["metadata"], indent=2))
