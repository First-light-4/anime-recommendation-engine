"""Optional saved-validation replay: never downloads, fits or overwrites artifacts.

Synthetic contract coverage lives in existing tests. These integration cases skip
explicitly on a clean clone without the ignored ARE-008 local artifacts.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from anime_recommender.evaluation.ranking_evaluator import evaluate_models, model_fingerprint
from anime_recommender.evaluation.recommendation_checks import evaluate_validation
from anime_recommender.evaluation.reports import load_report
from anime_recommender.ingestion.training_data import file_sha256
from anime_recommender.training.artifacts import load_models, mapping_fingerprint
from anime_recommender.training.inputs import load_training_inputs

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = {
    "1k": (772, 2091, 48080, .283710, .388166, .291948),
    "5k": (4621, 5279, 359967, .301667, .423164, .302479),
    "10k": (9455, 6719, 745354, .305145, .425284, .305863),
}


@pytest.mark.parametrize("version", ["1k", "5k", "10k"])
def test_saved_validation_replay_and_legacy_bundle_contract(version, monkeypatch):
    registry = json.loads((ROOT / "configs/model_versions.json").read_text(encoding="utf-8"))
    entry = registry["versions"][version]
    model_dir, data_dir = ROOT / entry["model_dir"], ROOT / entry["dataset_dir"]
    report_dir = ROOT / registry["comparison_dir"] / version
    prerequisites = (model_dir / "metadata.json", data_dir / "dataset_report.json", report_dir / "report.json")
    if not all(path.is_file() for path in prerequisites):
        pytest.skip(f"Optional local ARE-008 {version} artifacts absent; no automatic download/rebuild")
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("No network")))
    monkeypatch.setattr("implicit.cpu.als.AlternatingLeastSquares.fit", Mock(side_effect=AssertionError("No fit")))
    monkeypatch.setattr("anime_recommender.ingestion.training_data.acquire", Mock(side_effect=AssertionError("No ingestion")))
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert not path.name.startswith("test_"), "Validation replay read test split"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    protected = {p: file_sha256(p) for folder in (model_dir, report_dir)
                 for p in folder.iterdir() if p.is_file()}
    inputs = load_training_inputs(data_dir)
    bundle = load_models(model_dir, inputs.items, inputs.train_fingerprint)
    saved = load_report(report_dir)
    assert bundle.metadata["format_version"] == 1
    assert mapping_fingerprint(bundle.items) == bundle.metadata["mapping_fingerprint"]
    assert model_fingerprint(bundle) == saved.metadata["model_fingerprint"]
    if version == "1k":
        assert "model_version" not in bundle.metadata  # legacy ARE-006 stays supported
    else:
        assert bundle.metadata["model_version"] == version
    users, items, nnz, recall, ndcg, warm = PUBLISHED[version]
    assert inputs.matrix.shape == (users, items) and inputs.matrix.nnz == nnz
    before_items, before_users = bundle.als.item_factors.copy(), bundle.als.user_factors.copy()
    actual = evaluate_models(bundle, inputs, ks=saved.metadata["ks"],
                             include_als_fallback=saved.metadata["include_als_fallback"])
    pd.testing.assert_frame_equal(actual.tables["metrics"], saved.tables["metrics"],
                                  check_exact=False, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(actual.tables["recommendations"], saved.tables["recommendations"],
                                  check_exact=False, atol=1e-6, rtol=1e-5)
    ids = ["dataset_user_id", "model_name", "rank", "dataset_anime_id", "myanimelist_id"]
    pd.testing.assert_frame_equal(actual.tables["recommendations"][ids], saved.tables["recommendations"][ids])
    full = actual.tables["metrics"].query('model_name == "als" and mode == "full"').iloc[0]
    canonical_warm = actual.tables["metrics"].query('model_name == "als" and mode == "warm"').iloc[0]
    assert full.recall == pytest.approx(recall, abs=5e-7, rel=0)
    assert full.ndcg == pytest.approx(ndcg, abs=5e-7, rel=0)
    assert canonical_warm.recall == pytest.approx(warm, abs=5e-7, rel=0)
    # Verify the historical cohort, rather than inferring equality from rounded means.
    cohort = actual.tables["users"]
    assert len(cohort) == 100 and cohort.warm_context.gt(0).all() and cohort.positive_targets.gt(0).all()
    assert cohort.positive_targets.sub(cohort.cold_targets).gt(0).all()
    legacy = evaluate_validation(bundle.recommenders(), inputs.validation_context,
        inputs.observed_context, inputs.validation_target, inputs.validation_users, inputs.items, 10)
    old = legacy["summary"].query('model_name == "als"').iloc[0]
    assert old.warm_recall_users == canonical_warm.recall_eligible_users == 100
    assert old.warm_recall_at_k == pytest.approx(canonical_warm.recall, abs=1e-12)
    np.testing.assert_array_equal(before_items, bundle.als.item_factors)
    np.testing.assert_array_equal(before_users, bundle.als.user_factors)
    assert protected == {p: file_sha256(p) for p in protected}
