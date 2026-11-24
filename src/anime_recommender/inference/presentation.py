"""DataFrame presentation shared by inference and offline research; no IO/fit."""
import pandas as pd
from anime_recommender.inference.recommendation_table import ENGLISH_FIELDS, RUSSIAN_FIELDS, recommendation_table

def annotate_items(frame, workspace):
    """Keep original fields; attach certified titles and train-only popularity."""
    titles = list(dict.fromkeys(["title", *ENGLISH_FIELDS, *RUSSIAN_FIELDS]))
    columns = ["dataset_anime_id", *[name for name in titles if name in workspace.catalog]]
    return frame.merge(workspace.catalog[columns], on="dataset_anime_id",
                       how="left", validate="many_to_one").merge(
        workspace.bundle.popularity[["dataset_anime_id", "unique_users"]], on="dataset_anime_id",
        how="left", validate="many_to_one")

def profile_display(profile, nickname, show_history=False):
    """Presentation-only data, no HTTP, signal preparation or ranking duplication.

    No public YummyAnime catalog mapping/URLs are available for recommended items;
    never construct a YummyAnime URL from an unrelated MAL or dataset ID.
    """
    if profile["state"] != "ok":
        return {"summary": pd.Series({"State": profile["state"], "Error": profile["error"]}),
                "recommendations": pd.DataFrame(), "history": pd.DataFrame()}
    history = profile["history"]
    coverage = profile["coverage"]
    recs = profile["recommendations"]
    used = "popularity" if profile["fallback"] else "als"
    summary = pd.Series({
        "Nickname": nickname,
        "Anime total": history.anime_id.nunique(),
        "Completed": history.loc[history.list_id.eq(2), "anime_id"].nunique(),
        "Favorites": history.loc[history.is_fav.fillna(False), "anime_id"].nunique(),
        "Rated": history.loc[history.user_rating.between(1, 10).fillna(False), "anime_id"].nunique(),
        "Positive interactions": history.loc[history.positive, "anime_id"].nunique(),
        "Warm positives for fold-in": coverage["fold_in_interactions"],
        "MAL coverage (catalog)": coverage["catalog_coverage"],
        "MAL coverage (train vocabulary)": coverage["training_coverage"],
        "Matched train items": coverage["training_matches"],
        "Excluded candidate items": history.loc[history.known_in_training, "dataset_anime_id"].nunique(),
        "Model": used, "Fallback reason": profile["fallback_reason"],
    })
    table = recommendation_table(recs)
    return {"summary": summary, "recommendations": table,
            "history": history if show_history else history.iloc[:0].copy()}
