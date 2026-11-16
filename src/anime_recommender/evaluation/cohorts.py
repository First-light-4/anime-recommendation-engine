"""Cohorts depend only on distinct observed positive context and train items."""

import pandas as pd


COHORTS = ("0", "1", "2–4", "5–9", "10–19", "20+")


def cohort_label(length):
    if type(length) is not int or length < 0:
        raise ValueError("History length must be a nonnegative integer")
    for upper, label in zip((0, 1, 4, 9, 19), COHORTS):
        if length <= upper:
            return label
    return COHORTS[-1]


def assign_cohorts(user_mapping, positive_context, item_mapping):
    vocabulary = set(item_mapping.dataset_anime_id)
    context = {int(uid): set(group.dataset_anime_id) for uid, group in positive_context.groupby("dataset_user_id")}
    rows = []
    for uid in user_mapping.dataset_user_id:
        ids = context.get(int(uid), set())
        warm = len(ids & vocabulary)
        rows.append({"dataset_user_id": int(uid), "positive_context": len(ids), "warm_context": warm,
            "cold_context": len(ids) - warm, "cohort": cohort_label(warm)})
    return pd.DataFrame(rows)


def history_distribution(users):
    counts = users.cohort.value_counts().reindex(COHORTS, fill_value=0)
    return counts.rename_axis("cohort").to_frame("users").reset_index()
