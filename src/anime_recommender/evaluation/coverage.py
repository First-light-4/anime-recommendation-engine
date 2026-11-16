"""Catalog exposure and context/target coverage with explicit denominators."""


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def catalog_coverage(recommended_ids, vocabulary):
    ids, known = set(recommended_ids), set(vocabulary)
    if not ids <= known:
        raise ValueError("Recommendations outside train vocabulary")
    return {"recommended_items": len(ids), "catalog_items": len(known),
            "catalog_coverage": ratio(len(ids), len(known))}


def context_target_coverage(users):
    n = len(users)
    context = int(users.positive_context.sum())
    warm = int(users.warm_context.sum())
    targets = int(users.positive_targets.sum())
    cold_targets = int(users.cold_targets.sum())
    return {"users": n, "positive_context": context, "warm_context": warm,
        "cold_context": context - warm, "warm_context_coverage": ratio(warm, context),
        "cold_context_share": ratio(context - warm, context), "positive_targets": targets,
        "warm_targets": targets - cold_targets, "cold_targets": cold_targets,
        "positive_targets_mean": float(users.positive_targets.mean()) if n else None,
        "positive_targets_median": float(users.positive_targets.median()) if n else None,
        "warm_targets_mean": float((users.positive_targets - users.cold_targets).mean()) if n else None,
        "warm_targets_median": float((users.positive_targets - users.cold_targets).median()) if n else None,
        "warm_target_share": ratio(targets - cold_targets, targets),
        "cold_target_share": ratio(cold_targets, targets),
        "users_without_warm_context": int(users.warm_context.eq(0).sum()),
        "users_without_warm_context_share": ratio(int(users.warm_context.eq(0).sum()), n),
        "users_without_positive_targets": int(users.positive_targets.eq(0).sum()),
        "users_without_positive_targets_share": ratio(int(users.positive_targets.eq(0).sum()), n),
        "als_fold_in_users": int(users.warm_context.gt(0).sum()),
        "als_fold_in_share": ratio(int(users.warm_context.gt(0).sum()), n)}
