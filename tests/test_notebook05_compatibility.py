"""Optional real-bundle Notebook smoke, synthetic profile and zero network/fit."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def test_notebook05_saved_models_switch_without_side_effects():
    import json
    registry = json.loads((ROOT / "configs/model_versions.json").read_text(encoding="utf-8"))
    if not all((ROOT / entry["model_dir"] / "metadata.json").is_file() for entry in registry["versions"].values()):
        pytest.skip("Optional real bundles absent; notebook smoke never downloads or trains")
    from anime_recommender.ingestion.training_data import file_sha256
    path = ROOT / "notebooks/05_yummyanime_recommendations.ipynb"
    before = file_sha256(path)
    notebook = nbformat.read(path, as_version=4)
    setup = r'''
from pathlib import Path
from unittest.mock import Mock, patch
import os
from anime_recommender.inference import profile_session as session_module
OriginalSession = session_module.ProfileSession
fake = Mock(request_log=[])
fake.get_user.return_value = {"id": 123}
def source(mid, status):
    return {"anime_id": 100000+mid, "title": "Synthetic", "remote_ids": {"myanimelist_id": mid},
            "user": {"rating": 0, "list": {"list": {"id": status}, "is_fav": False}}}
fake.get_lists.return_value = [source(16498, 2), source(25777, 1), source(35760, 0)]
fake.get_favorites.return_value = []
session_module.ProfileSession = lambda root: OriginalSession(root, client_factory=lambda: fake)
title_lookup = Mock(side_effect=lambda profile, env: {**profile, "title_cache": {}})
session_module.enrich_profile_titles = title_lookup
guards = [patch("requests.sessions.Session.request", side_effect=AssertionError("No HTTP")),
          patch("implicit.cpu.als.AlternatingLeastSquares.fit", side_effect=AssertionError("No fit")),
          patch("anime_recommender.ingestion.training_data.acquire", side_effect=AssertionError("No ingestion")),
          patch("anime_recommender.training.dataset.run_pipeline", side_effect=AssertionError("No dataset rebuild"))]
for guard in guards: guard.start()
original_open = Path.open
def guarded_open(path, *args, **kwargs):
    if path.name.startswith("test_") or path.name in ("clean_interactions.parquet", "sample_interactions.parquet", "train_interactions.parquet", "train_matrix.npz"):
        raise AssertionError("Notebook read source/train/test interactions")
    return original_open(path, *args, **kwargs)
file_guard = patch.object(Path, "open", guarded_open)
file_guard.start()
os.environ["YUMMYANIME_TEST_NICKNAME"] = "SyntheticCompatibilityProfile"
'''
    assertions = r'''
from pandas.testing import assert_frame_equal
from anime_recommender.training.artifacts import mapping_fingerprint
assert model_version == "10k" and top_k == 15
source_history = profile_session._sources[nickname]["history"].copy(deep=True)
for version, count in (("1k",2091),("10k",6719),("5k",5279),("1k",2091)):
    result = profile_session.recommend(nickname, version, 15)
    profile_session.initial_titles(result, nickname)
    selected = profile_session.workspace(version)
    assert len(selected.bundle.items) == count
    assert mapping_fingerprint(selected.bundle.items) == selected.bundle.metadata["mapping_fingerprint"]
    if version != "1k": assert selected.bundle.metadata["model_version"] == version
    assert len(result["recommendations"]) == 15
    assert not set(result["recommendations"].dataset_anime_id) & set(result["signals"].seen_ids)
    assert result["franchise_diagnostics"]["enabled"]
    if result["model_franchise_coverage"]["covered_candidate_share"] < 1:
        assert result["model_franchise_coverage"]["warning"]
    assert_frame_equal(source_history, profile_session._sources[nickname]["history"])
assert len(profile_session.recommend(nickname, "10k", 7)["recommendations"]) == 7
fake.get_user.assert_called_once()
fake.get_lists.assert_called_once()
fake.get_favorites.assert_called_once()
title_lookup.assert_called_once()
'''
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    notebook.cells.append(nbformat.v4.new_code_cell(assertions))
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    assert file_sha256(path) == before
    saved = nbformat.read(path, as_version=4)
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for c in saved.cells if c.cell_type == "code")
