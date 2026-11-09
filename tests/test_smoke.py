"""Verify that the installed package exposes all component packages."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "anime_recommender",
        "anime_recommender.ingestion",
        "anime_recommender.preprocessing",
        "anime_recommender.training",
        "anime_recommender.evaluation",
        "anime_recommender.inference",
        "anime_recommender.api",
    ],
)
def test_package_import(module_name: str) -> None:
    module = import_module(module_name)

    assert module.__name__ == module_name
    assert module.__path__
