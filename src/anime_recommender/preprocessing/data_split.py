"""Stable user-disjoint allocation, train-only k-core and fold-in holdouts."""

import hashlib
import math

import pandas as pd

from anime_recommender.preprocessing.training_data import DataValidationError


def _priority(seed: int, domain: str, *ids) -> bytes:
    return hashlib.sha256(":".join(map(str, [seed, domain, *ids])).encode("ascii")).digest()


def assign_users(user_ids, seed=42, fractions=(.8, .1, .1)) -> pd.DataFrame:
    """Exact proportions with deterministic largest-remainder rounding.

    Assignment is before eligibility filtering, based on IDs only, not items/ratings.
    It is invariant to input row order. Changing the user population can change it.
    """
    if len(fractions) != 3 or any(not math.isfinite(f) or f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise DataValidationError("User split fractions must be positive and sum to 1")
    ids = sorted(set(int(value) for value in user_ids), key=lambda uid: (_priority(seed, "users", uid), uid))
    exact = [len(ids) * f for f in fractions]
    sizes = [math.floor(value) for value in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))
    for i in order[:len(ids) - sum(sizes)]:
        sizes[i] += 1
    labels = [label for label, size in zip(["train", "validation", "test"], sizes) for _ in range(size)]
    return pd.DataFrame({"dataset_user_id": pd.Series(ids, dtype="int64"), "split": labels}).sort_values("dataset_user_id").reset_index(drop=True)


def eligible_positives(positives, assignments, split, minimum):
    candidates = positives.loc[positives["dataset_user_id"].isin(
        assignments.loc[assignments["split"].eq(split), "dataset_user_id"])]
    counts = candidates.groupby("dataset_user_id").size()
    return candidates.loc[candidates["dataset_user_id"].isin(counts.index[counts.ge(minimum)])].copy()


def filter_train(positives: pd.DataFrame, min_user: int = 5,
                 min_item: int = 5) -> tuple[pd.DataFrame, list[dict]]:
    """Converge to a train-only bipartite k-core; no eval data accepted here."""
    frame = positives.copy()
    rounds = []
    while True:
        before = len(frame)
        users = frame.groupby("dataset_user_id").size()
        frame = frame.loc[frame["dataset_user_id"].isin(users.index[users.ge(min_user)])]
        user_removed = before - len(frame)
        items = frame.groupby("dataset_anime_id")["dataset_user_id"].nunique()
        retained = items.index[items.ge(min_item)]
        after_users = len(frame)
        frame = frame.loc[frame["dataset_anime_id"].isin(retained)]
        rounds.append({"iteration": len(rounds) + 1, "user_filter_removed_rows": user_removed,
                       "item_filter_removed_rows": after_users - len(frame), "remaining_rows": len(frame)})
        if len(frame) == before:
            break
    return frame.sort_values(["dataset_user_id", "dataset_anime_id"]).reset_index(drop=True), rounds


def split_observed_interactions(interactions: pd.DataFrame, seed=42,
                                context_fraction=.8, min_context=1, min_target=1):
    """Split raw cleaned history, independent of ratings and train vocabulary.

    Short histories remain in the cohort. If both raw minima cannot be satisfied,
    keep available observations in context and leave target empty, never drop user.
    """
    if not 0 < context_fraction < 1 or min_context < 1 or min_target < 1:
        raise DataValidationError("Invalid context/target settings")
    contexts, targets = [], []
    for user_id, group in interactions.groupby("dataset_user_id", sort=True):
        group = group.reset_index(drop=True)
        order = sorted(group.index, key=lambda i: (
            _priority(seed, "context", int(user_id), int(group.loc[i, "dataset_anime_id"])),
            int(group.loc[i, "dataset_anime_id"])))
        count = (len(group) if len(group) < min_context + min_target else
                 max(min_context, min(math.floor(len(group) * context_fraction), len(group) - min_target)))
        contexts.append(group.loc[order[:count]])
        targets.append(group.loc[order[count:]])
    def collect(parts):
        frame = pd.concat(parts, ignore_index=True) if parts else interactions.iloc[:0].copy()
        frame = frame.sort_values(["dataset_user_id", "dataset_anime_id"]).reset_index(drop=True)
        return frame
    return collect(contexts), collect(targets)


def split_context_target(interactions: pd.DataFrame, item_ids, seed=42,
                         context_fraction=.8, min_context=1, min_target=1):
    """Compatibility wrapper; the v2 pipeline calls the raw split directly."""
    parts = split_observed_interactions(interactions, seed, context_fraction, min_context, min_target)
    for frame in parts:
        frame["is_cold_item"] = ~frame["dataset_anime_id"].isin(item_ids)
    return parts
