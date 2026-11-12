"""Leakage and sparse-artifact invariants; no model fitting or target leakage."""

import pandas as pd

from anime_recommender.preprocessing.training_data import DataValidationError


def _check(condition, message):
    if not condition:
        raise DataValidationError(message)


def _pairs(frame):
    return set(zip(frame["dataset_user_id"], frame["dataset_anime_id"]))


def validate_dataset(train, evaluations, assignments, user_mapping, item_mapping, matrix,
                     expected_evaluations, min_user=5, min_item=5,
                     raw_evaluations=None, raw_cohorts=None, threshold=5):
    """Verify every held-out positive is retained, including cold targets."""
    _check(not assignments["dataset_user_id"].duplicated().any(), "A user has multiple split assignments")
    assigned = {name: set(assignments.loc[assignments["split"].eq(name), "dataset_user_id"])
                for name in ("train", "validation", "test")}
    _check(not (assigned["train"] & assigned["validation"] or assigned["train"] & assigned["test"]
                or assigned["validation"] & assigned["test"]), "Split user leakage")
    _check(set(train["dataset_user_id"]) <= assigned["train"], "Non-train user in train")
    _check(len(_pairs(train)) == len(train), "Duplicate train pairs")
    _check(train.groupby("dataset_user_id").size().ge(min_user).all(), "Train user minimum violated")
    _check(train.groupby("dataset_anime_id")["dataset_user_id"].nunique().ge(min_item).all(), "Train item minimum violated")
    _check(set(item_mapping["dataset_anime_id"]) == set(train["dataset_anime_id"]), "Item vocabulary is not exactly train-only")
    _check(set(user_mapping["dataset_user_id"]) == set(train["dataset_user_id"]), "Train user mapping mismatch")
    for mapping, id_col, index_col in [(user_mapping, "dataset_user_id", "matrix_row"),
                                      (item_mapping, "dataset_anime_id", "matrix_col")]:
        _check(not mapping[id_col].duplicated().any(), "Duplicate mapping ID")
        _check(mapping[id_col].tolist() == sorted(mapping[id_col]), "Unstable mapping order")
        _check(mapping[index_col].tolist() == list(range(len(mapping))), "Non-contiguous mapping indices")
    _check(not item_mapping["myanimelist_id"].duplicated().any() and item_mapping["myanimelist_id"].notna().all(), "Ambiguous MAL mapping")
    _check(matrix.format == "csr" and matrix.shape == (len(user_mapping), len(item_mapping)), "CSR dimensions mismatch")
    _check(matrix.nnz == len(train) and (matrix.data == 1).all(), "CSR nnz/confidence mismatch")
    coo = matrix.tocoo()
    observed = pd.DataFrame({"matrix_row": coo.row, "matrix_col": coo.col})
    expected = train[["dataset_user_id", "dataset_anime_id"]].merge(user_mapping).merge(item_mapping)
    _check(set(zip(observed["matrix_row"], observed["matrix_col"])) ==
           set(zip(expected["matrix_row"], expected["matrix_col"])), "CSR coordinates mismatch")
    vocabulary = set(item_mapping["dataset_anime_id"])
    checks = {"user_disjoint": True, "train_only_vocabulary": True, "csr_coordinates_valid": True}
    for name, (context, target) in evaluations.items():
        _check(set(context["dataset_user_id"]) <= assigned[name], f"Foreign user in {name} context")
        _check(set(target["dataset_user_id"]) <= assigned[name], f"Foreign user in {name} target")
        _check(not _pairs(context) & _pairs(target), f"Context-target leakage in {name}")
        _check(len(_pairs(context)) == len(context) and len(_pairs(target)) == len(target), f"Duplicate evaluation pairs in {name}")
        _check(_pairs(context) | _pairs(target) == _pairs(expected_evaluations[name]), f"Dropped or invented {name} positives/cold items")
        for frame in (context, target):
            _check(frame["is_cold_item"].eq(~frame["dataset_anime_id"].isin(vocabulary)).all(), "Incorrect cold-item flag")
        checks[name + "_context_target_disjoint"] = True
        checks[name + "_all_positive_rows_retained"] = True
        if raw_evaluations is not None:
            from anime_recommender.preprocessing.implicit_feedback import positive_feedback
            raw_context, raw_target = raw_evaluations[name]
            _check(not _pairs(raw_context) & _pairs(raw_target), f"Raw context-target leakage in {name}")
            _check(_pairs(raw_context) | _pairs(raw_target) == _pairs(raw_cohorts[name]), f"Dropped raw history in {name}")
            _check(_pairs(context) == _pairs(positive_feedback(raw_context, threshold)), f"Positive context reassigned from hidden target in {name}")
            _check(_pairs(target) == _pairs(positive_feedback(raw_target, threshold)), f"Positive target reassigned in {name}")
            checks[name + "_raw_holdout_complete"] = True
            checks[name + "_positives_follow_raw_split"] = True
    return checks


def evaluation_summary(context, target, cohort_users=None):
    combined = pd.concat([context, target], ignore_index=True)
    users = set(combined["dataset_user_id"]) if cohort_users is None else set(cohort_users)
    positive_context_users = set(context["dataset_user_id"])
    positive_target_users = set(target["dataset_user_id"])
    known_context = set(context.loc[~context["is_cold_item"], "dataset_user_id"])
    known_targets = set(target.loc[~target["is_cold_item"], "dataset_user_id"])
    anime = int(combined["dataset_anime_id"].nunique())
    cold_anime = int(combined.loc[combined["is_cold_item"], "dataset_anime_id"].nunique())
    def describe(frame):
        return {"rows": len(frame), "cold_rows": int(frame["is_cold_item"].sum()),
                "cold_share": float(frame["is_cold_item"].mean()) if len(frame) else None}
    return {"users": len(users), "anime": anime, "cold_anime": cold_anime,
            "cold_anime_share": cold_anime / anime if anime else None,
            "positive_rows": len(combined), "context": describe(context), "target": describe(target),
            "cold_interaction_share": float(combined["is_cold_item"].mean()) if len(combined) else None,
            "users_without_known_context": len(users - known_context),
            "users_without_positive_context": len(users - positive_context_users),
            "users_without_positive_targets": len(users - positive_target_users),
            "users_positive_targets_all_cold": len(positive_target_users - known_targets),
            "full_recall_users": len(positive_target_users),
            "warm_recall_users": len(known_context & known_targets),
            "users_without_known_target": len(users - known_targets),
            "users_with_known_context_and_target": len(known_context & known_targets)}


def evaluation_user_mapping(user_ids, raw_context, raw_target, context, target):
    """Account for every assigned user, even with no positive or warm rows."""
    users = pd.DataFrame({"dataset_user_id": pd.Series(sorted(user_ids), dtype="int64")})
    users["matrix_row"] = range(len(users))
    for name, frame in [("observed_context", raw_context), ("observed_target", raw_target),
                        ("positive_context", context), ("positive_target", target),
                        ("warm_context", context.loc[~context["is_cold_item"]]),
                        ("warm_target", target.loc[~target["is_cold_item"]])]:
        counts = frame.groupby("dataset_user_id").size()
        users[name + "_count"] = users["dataset_user_id"].map(counts).fillna(0).astype("int64")
    users["full_recall_eligible"] = users["positive_target_count"].gt(0)
    users["warm_recall_eligible"] = users["warm_context_count"].gt(0) & users["warm_target_count"].gt(0)
    return users


def validate_context_matrix(context, users, items, matrix):
    """Prove that only warm context coordinates enter the saved fold-in input."""
    warm = context.loc[~context["is_cold_item"]]
    coo = matrix.tocoo()
    row = warm["dataset_user_id"].map(users.set_index("dataset_user_id")["matrix_row"])
    col = warm["dataset_anime_id"].map(items.set_index("dataset_anime_id")["matrix_col"])
    _check(matrix.format == "csr" and matrix.shape == (len(users), len(items)), "Context CSR shape mismatch")
    _check(matrix.nnz == len(warm) and (matrix.data == 1).all(), "Context CSR nnz/confidence mismatch")
    _check(set(zip(coo.row, coo.col)) == set(zip(row, col)), "Context CSR includes non-context or cold coordinates")
