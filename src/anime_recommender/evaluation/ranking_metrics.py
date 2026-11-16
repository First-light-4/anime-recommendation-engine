"""Binary ordered ranking metrics. Empty targets are undefined for every metric.

Precision denominator is K even for short lists. AP denominator is min(K, |R|).
Recall uses all relevant items; NDCG ideal includes min(K, |R|) relevant items.
Duplicates anywhere in the recommendation list are rejected, never deduplicated.
"""

import math


METRICS = ("precision", "recall", "ndcg", "map", "hit_rate")


def ranking_metrics(recommendations, targets, k):
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer")
    ids = list(recommendations)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate recommendations")
    relevant = set(targets)
    if not relevant:
        return dict.fromkeys(METRICS)
    hits = 0
    dcg = ap = 0.0
    for rank, item in enumerate(ids[:k], 1):
        if item in relevant:
            hits += 1
            dcg += 1 / math.log2(rank + 1)
            ap += hits / rank
    ideal_count = min(k, len(relevant))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {"precision": hits / k, "recall": hits / len(relevant), "ndcg": dcg / ideal,
            "map": ap / ideal_count, "hit_rate": float(hits > 0)}


def precision_at_k(recommendations, targets, k):
    return ranking_metrics(recommendations, targets, k)["precision"]


def recall_at_k(recommendations, targets, k):
    return ranking_metrics(recommendations, targets, k)["recall"]


def ndcg_at_k(recommendations, targets, k):
    return ranking_metrics(recommendations, targets, k)["ndcg"]


def map_at_k(recommendations, targets, k):
    return ranking_metrics(recommendations, targets, k)["map"]


def hit_rate_at_k(recommendations, targets, k):
    return ranking_metrics(recommendations, targets, k)["hit_rate"]
