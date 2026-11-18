"""Bootstrap/exposure tests with hand-checkable data, no fits or HTTP."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from anime_recommender.evaluation.model_selection import (
    aligned_full_metrics, paired_bootstrap, popularity_exposure,
)
from anime_recommender.preprocessing.training_data import DataValidationError


def test_paired_constant_gain_alignment_and_determinism():
    baseline = pd.Series([.2, .4, .6], index=[1, 2, 3])
    improved = pd.Series([.7, .3, .5], index=[3, 1, 2])
    result = paired_bootstrap(baseline, improved)
    assert result == paired_bootstrap(baseline, improved)
    assert result["absolute_delta"] == pytest.approx(.1)
    assert result["relative_delta"] == pytest.approx(.25)
    # A model-independent bootstrap would invent uncertainty for this paired shift.
    assert result["ci95_low"] == pytest.approx(.1)
    assert result["ci95_high"] == pytest.approx(.1)
    assert result["improved_share"] == 1
    assert result["eligible_users"] == 3


def test_paired_undefined_zero_reference_and_ties():
    a = pd.Series([0., np.nan, .3, .5], index=[1, 2, 3, 4])
    b = pd.Series([.1, np.nan, .2, .5], index=a.index)
    result = paired_bootstrap(a, b)
    assert result["eligible_users"] == 3 and result["undefined_users"] == 1
    assert result["improved_share"] == pytest.approx(1/3)
    assert result["worsened_share"] == pytest.approx(1/3)
    assert result["unchanged_share"] == pytest.approx(1/3)
    identical = paired_bootstrap(a, a)
    assert identical["ci95_low"] == identical["ci95_high"] == 0
    assert identical["unchanged_share"] == 1
    zero = paired_bootstrap(pd.Series([0.]), pd.Series([.2]))
    assert zero["relative_delta"] is None
    empty = paired_bootstrap(pd.Series([np.nan]), pd.Series([np.nan]))
    assert empty["absolute_delta"] is None and empty["eligible_users"] == 0
    with pytest.raises(DataValidationError, match="masks"):
        paired_bootstrap(a, a.fillna(0))
    with pytest.raises(DataValidationError, match="IDs"):
        paired_bootstrap(a, a.iloc[:-1])
    with pytest.raises(DataValidationError, match="IDs"):
        paired_bootstrap(pd.Series([.1,.2], index=[1,1]), pd.Series([.1,.2], index=[1,1]))


def test_exposure_weights_occurrences_and_normalizes_train_size():
    items = pd.DataFrame({"dataset_anime_id": range(1, 11)})
    popularity = items.assign(unique_users=list(range(100, 0, -10)))
    recs = pd.DataFrame({"dataset_user_id": [1, 2, 3], "dataset_anime_id": [1, 1, 10], "rank": [1,1,1]})
    result, distribution = popularity_exposure(recs, popularity, items, 100)
    assert result["unique_recommended_items"] == 2
    assert result["catalog_coverage_10"] == .2
    assert result["mean_unique_train_users"] == 70
    assert result["mean_train_user_share"] == pytest.approx(.7)
    assert result["mean_popularity_percentile"] == 70
    assert result["top10_recommendation_share"] == pytest.approx(2/3)
    assert distribution.recommendations.sum() == 3
    assert distribution.share.sum() == pytest.approx(1)
    twice = popularity.assign(unique_users=popularity.unique_users*2)
    larger, _ = popularity_exposure(recs, twice, items, 200)
    assert larger["mean_unique_train_users"] == 140
    assert larger["mean_train_user_share"] == pytest.approx(result["mean_train_user_share"])
    assert larger["mean_popularity_percentile"] == result["mean_popularity_percentile"]


def test_exact_decile_ties_and_corrupt_popularity():
    items = pd.DataFrame({"dataset_anime_id": range(1, 12)})
    pop = items.assign(unique_users=1).iloc[::-1]
    recs = pd.DataFrame({"dataset_user_id": [1,2,3], "dataset_anime_id": [1,2,3], "rank": [1,1,1]})
    result, _ = popularity_exposure(recs, pop, items, 10)
    assert result["top10_item_count"] == 2
    assert result["top10_recommendation_share"] == pytest.approx(2/3)
    assert result["items_tied_at_boundary"] == 11
    with pytest.raises(DataValidationError, match="counts"):
        popularity_exposure(recs, pop.assign(unique_users=11), items, 10)
    with pytest.raises(DataValidationError, match="vocabulary"):
        popularity_exposure(recs.assign(dataset_anime_id=99), pop, items, 10)


def saved_reports():
    reports = {}
    for version in ("1k", "5k", "10k"):
        reports[version] = SimpleNamespace(
            metadata={"split":"validation", "franchise_filtering":False, "fixed_validation_fingerprint":"same"},
            tables={"users":pd.DataFrame({"dataset_user_id":[1,2], "positive_targets":[2,0]}),
                "user_metrics":pd.DataFrame({"dataset_user_id":[2,1], "model_name":["als"]*2,
                    "mode":["full"]*2, "k":[10]*2, "recall":[np.nan,.5], "ndcg":[np.nan,.7]}),
                "metrics":pd.DataFrame({"model_name":["als"], "mode":["full"], "k":[10], "recall":[.5], "ndcg":[.7]})})
    return reports


def test_saved_results_pairing_certificates_and_macro_consistency():
    reports = saved_reports()
    aligned = aligned_full_metrics(reports)
    assert aligned["1k"].index.tolist() == [1,2]
    assert pd.isna(aligned["10k"].loc[2,"recall"])
    corrupt = deepcopy(reports)
    corrupt["5k"].tables["metrics"].loc[0,"recall"] = .6
    with pytest.raises(DataValidationError, match="macro"):
        aligned_full_metrics(corrupt)
    corrupt = deepcopy(reports)
    corrupt["10k"].metadata["fixed_validation_fingerprint"] = "other"
    with pytest.raises(DataValidationError, match="fingerprint"):
        aligned_full_metrics(corrupt)
    corrupt = deepcopy(reports)
    corrupt["5k"].tables["users"].loc[0,"positive_targets"] = 3
    with pytest.raises(DataValidationError, match="target counts"):
        aligned_full_metrics(corrupt)
