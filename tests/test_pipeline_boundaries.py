"""Independent stages on tiny synthetic data; no remote requests or real fitting."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from anime_recommender.artifact_files import atomic_json, file_sha256
from anime_recommender.dataset_artifacts import TRAIN_FILES, VALIDATION_FILES
from anime_recommender.pipelines.evaluation import run_evaluation
from anime_recommender.pipelines.ingestion import run_ingestion
from anime_recommender.pipelines.preparation import run_cleaning, run_preparation
from anime_recommender.pipelines.training import run_training
from anime_recommender.train_inputs import load_train_view

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("No network")))
    source = json.loads((ROOT / "configs/training_data.json").read_text())
    source.update(sample_users=30, cache_dir="data/raw/synthetic", output_dir="data/processed/synthetic")
    cache = tmp_path / source["cache_dir"]
    cache.mkdir(parents=True)
    pd.DataFrame([(u, i, "8") for u in range(1, 31) for i in range(1, 9)],
                 columns=["userID", "animeID", "rating"]).to_parquet(cache / "sample_interactions.parquet", index=False)
    pd.DataFrame({"animeID": range(1, 9), "title": ["Synthetic"] * 8,
                  "mal_url": [f"https://myanimelist.net/anime/{i+100}" for i in range(1, 9)]
                 }).to_csv(cache / "anime_catalog.csv", index=False)
    atomic_json(cache / "manifest.json", {
        "parameters": {"source": source["source"], "sample_users": 30, "seed": source["seed"],
                       "sampling_version": "sha256-bottom-k-complete-users-v1"},
        "cache_sha256": {n: file_sha256(cache / n) for n in ("sample_interactions.parquet", "anime_catalog.csv")}})
    split = json.loads((ROOT / "configs/ml_dataset.json").read_text())
    split.update(input_dir=source["output_dir"], output_dir="data/processed/split")
    config = json.loads((ROOT / "configs/recommendation_models.json").read_text())
    config["als"].update(factors=2, iterations=2, num_threads=1)
    # Native factors are assigned, not fitted; test orchestration/publication only.
    def synthetic_factors(matrix, als_config, blas_threads):
        from implicit.cpu.als import AlternatingLeastSquares
        model = AlternatingLeastSquares(**{k: v for k, v in als_config.items() if k != "cg_steps"})
        model.cg_steps = als_config["cg_steps"]
        model.user_factors = np.ones((matrix.shape[0], 2), dtype="float32")
        model.item_factors = np.arange(matrix.shape[1] * 2, dtype="float32").reshape(-1, 2) / 10
        return model
    monkeypatch.setattr("anime_recommender.training.fitting.train_als", synthetic_factors)
    monkeypatch.setattr("implicit.cpu.als.AlternatingLeastSquares.fit", Mock(side_effect=AssertionError("No fit")))
    run_cleaning(source, tmp_path)
    run_preparation(split, tmp_path)
    return tmp_path, source, split, config


def test_preparation_reuses_identical_artifacts_without_fit_or_acquisition(local_data, monkeypatch):
    root, source, split, _ = local_data
    monkeypatch.setattr("anime_recommender.ingestion.training_data.acquire", Mock(side_effect=AssertionError("No acquire")))
    protected = {p: file_sha256(p) for p in (root / "data").rglob("*") if p.is_file()}
    assert run_ingestion(source, root)["mode"] == "reused"
    run_cleaning(source, root)
    prepared = run_preparation(split, root)
    assert prepared.report["config"]["positive_threshold"] == 5
    assert prepared.report["config"]["min_user_positives"] == 5
    assert protected == {p: file_sha256(p) for p in protected}
    wrong = {**split, "seed": split["seed"] + 1}
    with pytest.raises(ValueError, match="configuration differs"):
        run_preparation(wrong, root)
    assert protected == {p: file_sha256(p) for p in protected}


def test_training_requires_no_validation_and_executes_only_explicitly(local_data, monkeypatch):
    root, _, split, config = local_data
    directory = root / split["output_dir"]
    for path in directory.glob("*.parquet"):
        if path.name.startswith(("validation_", "test_")):
            path.unlink()
    for path in directory.glob("*_context_matrix.npz"):
        path.unlink()
    monkeypatch.setattr("anime_recommender.evaluation.ranking_evaluator.evaluate_models",
                        Mock(side_effect=AssertionError("No evaluation")))
    output = root / "artifacts/models/synthetic"
    plan = run_training(directory, config, output)
    assert not plan["execute"] and not output.exists()
    bundle = run_training(directory, config, output, execute=True)
    assert bundle.metadata["training_split"] == "train only"
    assert bundle.metadata["train_fingerprint"] == load_train_view(directory).train_fingerprint
    before = {p: file_sha256(p) for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        run_training(directory, config, output, execute=True)
    assert before == {p: file_sha256(p) for p in before}


def test_evaluation_requires_no_train_files_and_preserves_canonical_metrics(local_data):
    root, _, split, config = local_data
    directory, output = root / split["output_dir"], root / "artifacts/models/synthetic"
    run_training(directory, config, output, execute=True)
    before = run_evaluation(output, directory, {"ks": [10]})
    for name in TRAIN_FILES:
        (directory / name).unlink()
    report_dir = root / "artifacts/evaluation/synthetic"
    after = run_evaluation(output, directory, {"ks": [10]}, output_dir=report_dir)
    assert before.metadata["protocol_version"] == after.metadata["protocol_version"] == "ARE-007A-v1"
    assert_frame_equal(before.tables["metrics"], after.tables["metrics"])
    assert_frame_equal(before.tables["recommendations"], after.tables["recommendations"])
    with pytest.raises(FileExistsError):
        run_evaluation(output, directory, {"ks": [10]}, output_dir=report_dir)


def test_network_operations_require_explicit_permission(tmp_path):
    from anime_recommender.pipelines.franchise_metadata import run_franchise_metadata
    config = json.loads((ROOT / "configs/training_data.json").read_text())
    assert run_ingestion(config, tmp_path)["mode"] == "plan"
    with pytest.raises(ValueError, match="allow_network"):
        run_ingestion(config, tmp_path, execute=True)
    with pytest.raises(ValueError, match="allow_network"):
        run_franchise_metadata(tmp_path, execute=True)


def test_all_project_imports_including_deferred_have_no_cycles():
    root, graph, trees = ROOT / "src/anime_recommender", {}, {}
    for path in root.rglob("*.py"):
        name = "anime_recommender." + path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        name = name.removesuffix(".__init__")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trees[name] = tree
        graph[name] = set()
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("anime_recommender"):
                graph[name].add(node.module)
                for alias in node.names:
                    submodule = node.module + "." + alias.name
                    if submodule in trees:
                        graph[name].add(submodule)
            elif isinstance(node, ast.Import):
                graph[name].update(alias.name for alias in node.names
                                   if alias.name.startswith("anime_recommender"))
    active, finished = [], set()
    def visit(name):
        assert name not in active, "Deferred/direct cycle: " + " -> ".join(active + [name])
        if name in finished:
            return
        active.append(name)
        for target in graph.get(name, ()):
            visit(target)
        active.pop()
        finished.add(name)
    for name in graph:
        visit(name)


@pytest.mark.parametrize("entry,forbidden", [
    ("pipelines.training", ("evaluation", "validation_inputs", "training.pipeline", "ingestion")),
    ("pipelines.evaluation", ("training", "ingestion")),
    ("pipelines.preparation", ("training", "evaluation")),
    ("pipelines.ingestion", ("training", "evaluation", "preprocessing.dataset")),
    ("pipelines.inference", ("training", "evaluation", "ingestion.viewing_order_metadata")),
    ("pipelines.franchise_metadata", ("training", "evaluation", "inference")),
])
def test_stage_import_boundaries_in_fresh_interpreter(entry, forbidden):
    code = '''import importlib,importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==n or fullname.startswith(n+".") for n in FORBIDDEN):
   raise AssertionError("Forbidden stage dependency: "+fullname)
sys.meta_path.insert(0,Guard())
import requests
requests.sessions.Session.request=lambda *a,**k: (_ for _ in ()).throw(AssertionError("No network"))
importlib.import_module(ENTRY)
'''
    prefix = "FORBIDDEN=" + repr(tuple("anime_recommender." + n for n in forbidden)) + "\n"
    prefix += "ENTRY=" + repr("anime_recommender." + entry) + "\n"
    result = subprocess.run([sys.executable, "-c", prefix + code], cwd=ROOT,
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("version", ["1k", "5k", "10k"])
def test_load_only_evaluation_replays_saved_validation_without_train_reads(version, monkeypatch):
    from anime_recommender.evaluation.reports import load_report
    config = json.loads((ROOT / "configs/model_versions.json").read_text())
    entry = config["versions"][version]
    model_dir, data_dir = ROOT / entry["model_dir"], ROOT / entry["dataset_dir"]
    report_dir = ROOT / config["comparison_dir"] / version
    if not all(p.is_file() for p in (model_dir / "metadata.json", data_dir / "dataset_report.json", report_dir / "report.json")):
        pytest.skip("Optional saved artifacts absent; no rebuild/download")
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("No network")))
    monkeypatch.setattr("implicit.cpu.als.AlternatingLeastSquares.fit", Mock(side_effect=AssertionError("No fit")))
    original = pd.read_parquet
    def guarded(path, *args, **kwargs):
        path = Path(path)
        assert not path.name.startswith("test_")
        if path.resolve().is_relative_to(data_dir.resolve()):
            assert path.name in VALIDATION_FILES, "Evaluation read train tables"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    saved = load_report(report_dir)
    actual = run_evaluation(model_dir, data_dir, {"ks": saved.metadata["ks"],
                            "include_als_fallback": saved.metadata["include_als_fallback"]})
    assert_frame_equal(actual.tables["metrics"], saved.tables["metrics"],
                       check_exact=False, atol=1e-10, rtol=1e-10)
    assert_frame_equal(actual.tables["recommendations"], saved.tables["recommendations"],
                       check_exact=False, atol=1e-6, rtol=1e-5)
