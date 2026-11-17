"""Pure structured post-processing and bounded local candidate-pool expansion."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anime_recommender.inference.recommendations import OUTPUT_COLUMNS, context_ids


@dataclass(repr=False)
class FranchiseResult:
    recommendations: pd.DataFrame
    diagnostics: dict
    audit: pd.DataFrame


def filter_candidates(candidates, mapping, observed_mal_ids=(), completed_mal_ids=(),
                      observed_dataset_ids=(), top_k=15, max_per_franchise=1, train_mal_ids=None):
    for value in (top_k, max_per_franchise):
        if type(value) is not int or value < 1:
            raise ValueError("Recommendation and franchise limits must be positive integers")
    if candidates.dataset_anime_id.duplicated().any() or candidates.myanimelist_id.dropna().duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    if not candidates.score.is_monotonic_decreasing or not np.isfinite(candidates.score.to_numpy(dtype=float)).all():
        raise ValueError("Candidates must be finite and score-sorted")
    observed, completed, dataset_seen = set(context_ids(observed_mal_ids)), set(context_ids(completed_mal_ids)), set(context_ids(observed_dataset_ids))
    available = set(train_mal_ids) if train_mal_ids is not None else set(candidates.myanimelist_id.dropna())
    if not completed <= observed:
        raise ValueError("Completed anime must be part of observed history")
    earlier, effective_completed = set(), set()
    for mal in completed:
        ancestors = mapping.ancestors(mal)
        if ancestors is not None:
            earlier.update(ancestors)
            effective_completed.update(ancestors)
            effective_completed.add(mapping.groups[mal])
    selected, counts, audit = [], {}, []
    for position, row in enumerate(candidates.itertuples(index=False)):
        mal = None if pd.isna(row.myanimelist_id) else int(row.myanimelist_id)
        family = mapping.franchise.get(mal)
        ancestors = mapping.ancestors(mal)
        sequence = mapping.sequence_state(mal, completed)
        declared_sequence = next((s for s in mapping.snapshot.get("ordered_sequences", []) if s["franchise_id"] == family), None)
        unknown = family is None or (mal in mapping.mainline and ancestors is None) or (declared_sequence is not None and not declared_sequence["order_known"])
        if mal in observed or row.dataset_anime_id in dataset_seen:
            reason = "observed"
        elif sequence is not None and sequence["candidate_step"] is not None:
            stage = sequence["candidate_step"]
            if stage < sequence["last_completed_step"]:
                reason = "earlier_season"
            elif stage == sequence["last_completed_step"]:
                reason = "completed_stage"
            elif stage != sequence["next_step"]:
                reason = "next_part_unavailable" if not set(sequence["next_mal_ids"]) & available else "missing_prerequisite"
            else:
                reason = None
        elif sequence is not None and not sequence["finished"] and set(sequence["next_mal_ids"]) & available and not set(sequence["next_mal_ids"]) & observed:
            # Prioritize the actual next stage; optional parts are not called watched
            # or inserted as prerequisites, and remain eligible once progression ends.
            reason = "optional_deferred"
        elif mal in mapping.mainline and mapping.groups[mal] in earlier:
            reason = "earlier_season"
        elif ancestors is not None and not set(ancestors) <= effective_completed:
            reason = "missing_prerequisite"
        else:
            reason = None
        if reason is None and family is not None and counts.get(family, 0) >= max_per_franchise:
            reason = "franchise_duplicate"
        elif reason is None and len(selected) >= top_k:
            reason = "below_cutoff"
        elif reason is None:
            reason = "selected"
            selected.append(position)
            if family is not None:
                counts[family] = counts.get(family, 0) + 1
        audit.append({"dataset_anime_id": row.dataset_anime_id, "myanimelist_id": mal,
            "candidate_rank": row.rank, "franchise_id": family, "unknown_relationships": unknown, "reason": reason})
    result = candidates.iloc[selected][OUTPUT_COLUMNS].copy().reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    frame = pd.DataFrame(audit, columns=["dataset_anime_id", "myanimelist_id", "candidate_rank", "franchise_id", "unknown_relationships", "reason"])
    diagnostics = {"enabled": True, "candidates_before_filtering": len(candidates),
        "known_franchise_groups": len(set(mapping.franchise.values())),
        "covered_candidate_share": float(frame.franchise_id.notna().mean()) if len(frame) else None,
        "excluded_observed": int(frame.reason.eq("observed").sum()),
        "excluded_earlier_seasons": int(frame.reason.eq("earlier_season").sum()),
        "excluded_missing_prerequisite": int(frame.reason.eq("missing_prerequisite").sum()),
        "excluded_completed_stage": int(frame.reason.eq("completed_stage").sum()),
        "excluded_optional_deferred": int(frame.reason.eq("optional_deferred").sum()),
        "excluded_next_unavailable": int(frame.reason.eq("next_part_unavailable").sum()),
        "excluded_franchise_duplicates": int(frame.reason.eq("franchise_duplicate").sum()),
        "unknown_relationships": int(frame.unknown_relationships.sum()),
        "unknown_final_recommendations": int((frame.unknown_relationships & frame.reason.eq("selected")).sum()),
        "final_recommendations_count": len(result), "max_per_franchise": max_per_franchise,
        "relations_fingerprint": mapping.fingerprint}
    diagnostics["excluded_season_duplicates"] = diagnostics["excluded_franchise_duplicates"] + diagnostics["excluded_completed_stage"]
    diagnostics["excluded_unsuitable_continuations"] = diagnostics["excluded_earlier_seasons"] + diagnostics["excluded_missing_prerequisite"] + diagnostics["excluded_next_unavailable"]
    return FranchiseResult(result, diagnostics, frame)


def recommend_with_franchise(recommender, train_items, positive_context, mapping,
                             observed_dataset_ids=(), observed_mal_ids=(), completed_mal_ids=(),
                             top_k=15, max_per_franchise=1, candidate_multiplier=3):
    if any(type(v) is not int or v < 1 for v in (top_k, max_per_franchise, candidate_multiplier)):
        raise ValueError("Invalid franchise inference limits")
    n_items = len(train_items)
    if not n_items:
        raise ValueError("Empty train vocabulary")
    pool, attempts = min(n_items, top_k * candidate_multiplier), []
    while True:
        raw = recommender.recommend(positive_context, top_k=pool, exclude_seen=False,
                                    seen_items=observed_dataset_ids)
        filtered = filter_candidates(raw.recommendations, mapping, observed_mal_ids, completed_mal_ids,
            observed_dataset_ids, top_k, max_per_franchise, train_items.myanimelist_id.dropna())
        attempts.append(pool)
        if not raw.available or len(filtered.recommendations) >= top_k or pool == n_items:
            filtered.diagnostics.update({"candidate_pool_attempts": attempts, "pool_exhausted": pool == n_items,
                "insufficient_candidates": len(filtered.recommendations) < top_k,
                "model_name": recommender.model_name, "model_available": raw.available,
                "model_unavailable_reason": raw.reason})
            known = int(train_items.myanimelist_id.isin(mapping.franchise).sum())
            filtered.diagnostics.update({"known_franchise_train_items": known,
                "train_items": n_items, "franchise_train_coverage": known / n_items})
            return filtered
        pool = min(n_items, max(pool + 1, pool * 2))
