"""Read-only Notebook 07 smoke; no experiment execution, fit, network or test."""
from pathlib import Path
import json

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient
ROOT = Path(__file__).resolve().parents[1]


def test_notebook07_reads_existing_comparison_without_execution():
    from anime_recommender.artifact_files import file_sha256
    config = json.loads((ROOT / "configs/model_versions.json").read_text())
    if not (ROOT / config["comparison_dir"] / "comparison.json").is_file():
        pytest.skip("Optional saved comparison absent; no experiment execution")
    path = ROOT / "notebooks/07_training_data_scaling.ipynb"
    before = file_sha256(path)
    notebook = nbformat.read(path, as_version=4)
    setup = '''from unittest.mock import patch
from pathlib import Path
import pandas as pd
guards=[patch("requests.sessions.Session.request",side_effect=AssertionError("No network")),
        patch("implicit.cpu.als.AlternatingLeastSquares.fit",side_effect=AssertionError("No fit")),
        patch("anime_recommender.ingestion.training_data.acquire",side_effect=AssertionError("No acquisition"))]
for guard in guards:guard.start()
original=pd.read_parquet
def read_guard(path,*a,**k):
 assert not Path(path).name.startswith("test_"),"Notebook 07 read test"
 return original(path,*a,**k)
table_guard=patch.object(pd,"read_parquet",read_guard)
table_guard.start()
'''
    assertions = '''assert overview["summary"].train_items.tolist()==[2091,5279,6719]
assert overview["summary"].train_positives.tolist()==[48080,359967,745354]
assert comparison_result["metadata"]["test_used"] is False
from anime_recommender.registry.models import load_version_registry
assert load_version_registry(ROOT)["default_version"] == "10k"
'''
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    notebook.cells.append(nbformat.v4.new_code_cell(assertions))
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    assert file_sha256(path) == before
    saved = nbformat.read(path, as_version=4)
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in saved.cells if cell.cell_type == "code")
