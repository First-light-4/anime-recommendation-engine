"""Presentation tables for Notebook 06; no metric implementations or inference."""

from anime_recommender.evaluation.cohorts import COHORTS


def dashboard_tables(report, k=10):
    if k not in report.metadata["ks"]:
        raise ValueError("Dashboard K must be in evaluated K values")
    metrics = report.tables["metrics"]
    primary = metrics.loc[metrics.model_name.isin(["als", "popularity"])].copy()
    full = primary.loc[primary["mode"].eq("full")]
    both = primary.assign(series=primary.model_name + " / " + primary["mode"])
    cohorts = report.tables["cohorts"].loc[lambda f: f.model_name.isin(["als", "popularity"]) & f["mode"].eq("full") & f.k.eq(k)]
    coverage = report.tables["coverage"].loc[lambda f: f.model_name.isin(["als", "popularity"])]
    target_sizes = cohorts.loc[cohorts.model_name.eq("als"), ["cohort", "users",
        "positive_targets_mean", "positive_targets_median", "warm_targets_mean", "warm_targets_median"]]
    target_sizes = target_sizes.set_index("cohort").reindex(COHORTS)
    return {"main": metrics.loc[metrics["mode"].eq("full") & metrics.k.eq(k), ["model_name", "k",
                "precision", "recall", "ndcg", "map", "hit_rate", "users", *[m + "_eligible_users" for m in ("precision", "recall", "ndcg", "map", "hit_rate")]]],
        "recall_curve": full.pivot(index="k", columns="model_name", values="recall"),
        "ndcg_curve": full.pivot(index="k", columns="model_name", values="ndcg"),
        "full_warm_recall": both.pivot(index="k", columns="series", values="recall"),
        "catalog_coverage": coverage.pivot(index="k", columns="model_name", values="catalog_coverage"),
        "cohorts": cohorts,
        "cohort_target_sizes": target_sizes,
        "cohort_recall": cohorts.pivot(index="cohort", columns="model_name", values="recall").reindex(COHORTS),
        "cold_statistics": metrics.loc[metrics["mode"].eq("full") & metrics.k.eq(k), ["model_name", "users",
            "positive_context", "warm_context", "cold_context", "warm_context_coverage", "cold_context_share",
            "positive_targets", "warm_targets", "cold_targets", "cold_target_share",
            "users_without_warm_context", "users_without_positive_targets", "als_fold_in_share"]],
        "examples": report.tables["examples"], "example_users": report.tables["example_users"]}
