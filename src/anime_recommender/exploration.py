"""Small research orchestration layer; HTTP, normalization and checks stay separate."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from anime_recommender.ingestion.yummyanime import YummyAnimeClient, YummyAnimeError
from anime_recommender.preprocessing.user_data import normalize_user_lists
from anime_recommender.evaluation.user_data import compare_favorites, compare_rating_statistics


def project_root() -> Path:
    """Notebook launcher may start from the root or any directory within it."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/anime_recommender").is_dir():
            return candidate
    raise FileNotFoundError("Run from within the Anime Recommendation Engine project")


@dataclass(repr=False)
class ProfileExploration:
    # Suppress default repr: it would include a private profile and its history.
    profile: dict | None = None
    rows: list[dict] = field(default_factory=list)
    frame: pd.DataFrame = field(default_factory=lambda: normalize_user_lists([]))
    favorites: dict | None = None
    ratings: dict | None = None
    complete: bool = False
    error: str | None = None


def explore_profile(client: YummyAnimeClient, nickname: str,
                    cross_checks: bool = True) -> ProfileExploration:
    """Stop after the first API failure. Never save raw responses or retry."""
    result = ProfileExploration()
    try:
        result.profile = client.get_user(nickname)
        user_id = result.profile["id"]
        result.rows = client.get_lists(user_id)
        result.frame = normalize_user_lists(result.rows)
        if cross_checks:
            result.favorites = compare_favorites(result.frame, client.get_favorites(user_id))
            result.ratings = compare_rating_statistics(result.frame, client.get_rating_statistics(user_id))
        result.complete = True
    except YummyAnimeError as error:
        result.error = str(error)
    return result
