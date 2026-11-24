"""Small static guards for confirmed default/encoding/documentation regressions."""

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_documented_default_matches_registry_and_notebook05():
    registry = json.loads((ROOT / "configs/model_versions.json").read_text(encoding="utf-8"))
    assert registry["default_version"] == "10k"
    nb = json.loads((ROOT / "notebooks/05_yummyanime_recommendations.ipynb").read_text(encoding="utf-8"))
    values = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = node.value.value
    assert values["model_version"] == registry["default_version"]
    assert values["top_k"] == 15
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current = next(line for line in readme.splitlines() if "Текущая default-модель" in line)
    assert "10k" in current
    assert "Default остаётся `1k`" not in readme
    assert "docs/ARE-008-artifacts-and-reproduction.md" in readme


def test_are008_documents_are_utf8_and_local_links_exist():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("ARE-008*.md"))]:
        body = path.read_text(encoding="utf-8")
        assert "\ufffd" not in body
        assert not re.search(r"\?{3,}", body), f"Lost text in {path.name}"
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
            if target.startswith(("http:", "https:", "#")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path.name, target)
