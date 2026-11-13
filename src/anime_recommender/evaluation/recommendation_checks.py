"""ARE-006 validation diagnostics only. No test evaluation or tuning framework."""

import math

import numpy as np
import pandas as pd

from anime_recommender.inference.recommendations import context_ids, OUTPUT_COLUMNS
from anime_recommender.preprocessing.training_data import DataValidationError


def check_recommendations(result, item_mapping, seen, top_k):
    recs = result.recommendations
    vocabulary = set(item_mapping.dataset_anime_id)
    allowed = vocabulary - set(seen)
    expected = min(top_k, len(allowed)) if result.available else 0
    if list(recs.columns) != OUTPUT_COLUMNS or len(recs) != expected:
        raise DataValidationError("Incorrect recommendation schema/Top-N size")
    if not set(recs.dataset_anime_id) <= allowed or recs.dataset_anime_id.duplicated().any():
        raise DataValidationError("Unknown, seen or duplicate recommendation")
    if not np.isfinite(recs.score.to_numpy(dtype="float64")).all() or recs["rank"].tolist() != list(range(1, len(recs) + 1)):
        raise DataValidationError("Invalid score/rank")
    if not recs.score.is_monotonic_decreasing:
        raise DataValidationError("Recommendations are not score-sorted")
    expected_mal = recs.dataset_anime_id.map(item_mapping.set_index("dataset_anime_id").myanimelist_id)
    if expected_mal.tolist() != recs.myanimelist_id.tolist():
        raise DataValidationError("Recommendation MAL mapping mismatch")


def generate_validation_recommendations(recommenders, user_mapping, context, observed_context, items, top_k=10):
    """No target parameter: candidates/seen filtering are identical for both models."""
    histories = {int(uid): group.dataset_anime_id.tolist() for uid, group in context.groupby("dataset_user_id")}
    observed = {int(uid): group.dataset_anime_id.tolist() for uid, group in observed_context.groupby("dataset_user_id")}
    records, statuses = [], []
    for uid in user_mapping.dataset_user_id:
        uid = int(uid)
        positive = context_ids(histories.get(uid, []))
        seen = context_ids(observed.get(uid, []))
        for name, recommender in recommenders.items():
            result = recommender.recommend(positive, top_k=top_k, seen_items=seen)
            check_recommendations(result, items, sorted(set(seen) | set(positive)), top_k)
            recs = result.recommendations.copy()
            recs.insert(0, "dataset_user_id", uid)
            if not recs.empty:
                records.append(recs)
            statuses.append({"dataset_user_id": uid, "model_name": name, "als_available": result.available,
                             "recommendations": len(recs), **result.diagnostics})
    recommendations = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["dataset_user_id", *OUTPUT_COLUMNS])
    return recommendations, pd.DataFrame(statuses)


def diagnostic_metrics(recommendations, statuses, target, user_mapping, item_mapping, top_k=10):
    vocabulary = set(item_mapping.dataset_anime_id)
    targets = {int(uid): set(group.dataset_anime_id) for uid, group in target.groupby("dataset_user_id")}
    rows = []
    for status in statuses.to_dict("records"):
        uid, name = status["dataset_user_id"], status["model_name"]
        wanted = targets.get(uid, set())
        warm = wanted & vocabulary
        selected = recommendations.loc[recommendations.dataset_user_id.eq(uid) & recommendations.model_name.eq(name)].sort_values("rank")
        ids = selected.dataset_anime_id.tolist()
        hits = len(set(ids) & wanted)
        recall = hits / len(wanted) if wanted else None
        dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(ids, 1) if item in wanted)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(top_k, len(wanted)) + 1))
        warm_eligible = bool(warm) and status["warm_context"] > 0
        rows.append({"dataset_user_id": uid, "model_name": name, "full_recall": recall,
            "full_ndcg": dcg / idcg if wanted else None,
            "warm_recall": len(set(ids) & warm) / len(warm) if warm_eligible else None,
            "positive_targets": len(wanted), "cold_targets": len(wanted - vocabulary),
            "warm_context": status["warm_context"], "als_available": status["als_available"]})
    per_user = pd.DataFrame(rows)
    summaries = []
    for name, group in per_user.groupby("model_name", sort=False):
        recommended = set(recommendations.loc[recommendations.model_name.eq(name), "dataset_anime_id"])
        summaries.append({"model_name": name, "system": "ALS-only, no fallback" if name == "als" else "popularity",
            "users": len(group), "full_recall_users": int(group.full_recall.notna().sum()),
            "warm_recall_users": int(group.warm_recall.notna().sum()),
            "users_without_positive_targets": int(group.positive_targets.eq(0).sum()),
            "users_without_warm_context": int(group.warm_context.eq(0).sum()),
            "unavailable_users": int((~group.als_available).sum()),
            "recall_at_k": float(group.full_recall.mean()) if group.full_recall.notna().any() else None,
            "full_recall_at_k": float(group.full_recall.mean()) if group.full_recall.notna().any() else None,
            "ndcg_at_k": float(group.full_ndcg.mean()) if group.full_ndcg.notna().any() else None,
            "warm_recall_at_k": float(group.warm_recall.mean()) if group.warm_recall.notna().any() else None,
            "recommended_items": len(recommended), "catalog_coverage": len(recommended) / len(vocabulary) if vocabulary else None,
            "cold_targets": int(group.cold_targets.sum()), "positive_targets": int(group.positive_targets.sum()),
            "cold_target_share": float(group.cold_targets.sum() / group.positive_targets.sum()) if group.positive_targets.sum() else None})
    return pd.DataFrame(summaries), per_user


def evaluate_validation(recommenders, context, observed_context, target, user_mapping, item_mapping, top_k=10):
    if set(zip(observed_context.dataset_user_id, observed_context.dataset_anime_id)) & set(zip(target.dataset_user_id, target.dataset_anime_id)):
        raise DataValidationError("Observed-context / target leakage")
    recs, statuses = generate_validation_recommendations(recommenders, user_mapping, context, observed_context, item_mapping, top_k)
    summary, per_user = diagnostic_metrics(recs, statuses, target, user_mapping, item_mapping, top_k)
    return {"summary": summary, "per_user": per_user, "recommendations": recs, "statuses": statuses}
