"""Fresh-interpreter import boundaries: tests must not rely on pytest import order."""

import ast
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_inference_entrypoints_never_import_training_evaluation_or_research():
    code='''import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(("anime_recommender.training","anime_recommender.evaluation")) or fullname=="anime_recommender.recommendation_exploration":
   raise AssertionError("Forbidden serving dependency: "+fullname)
sys.meta_path.insert(0,Guard())
import requests
requests.sessions.Session.request=lambda *a,**k: (_ for _ in ()).throw(AssertionError("No network"))
from implicit.cpu.als import AlternatingLeastSquares
AlternatingLeastSquares.fit=lambda *a,**k: (_ for _ in ()).throw(AssertionError("No fit"))
import anime_recommender.contracts
import anime_recommender.artifact_io
import anime_recommender.registry.models
import anime_recommender.inference.model_versions
import anime_recommender.inference.profile_session
from pathlib import Path
from anime_recommender.registry.models import ModelRegistry
registry=ModelRegistry(Path.cwd())
assert registry.default_version=="10k"
# Real local bundles are optional on a clean clone; never fetch or rebuild them.
for version in ("1k","5k","10k"):
 if registry.is_available(version):
  bundle=registry.load(version)
  assert len(bundle.items)==len(bundle.als.item_factors)
assert not any(n.startswith(("anime_recommender.training","anime_recommender.evaluation")) for n in sys.modules)
'''
    result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr


def test_new_serving_modules_have_no_dependency_cycles():
    root=ROOT/"src/anime_recommender"
    nodes={}
    for path in root.rglob('*.py'):
        name='anime_recommender.'+path.relative_to(root).with_suffix('').as_posix().replace('/','.')
        name=name.removesuffix('.__init__')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes[name]={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('anime_recommender')}
    starts=['anime_recommender.artifact_io','anime_recommender.registry.models','anime_recommender.inference.profile_session','anime_recommender.inference.model_versions']
    active,finished=set(),set()
    def visit(name):
        assert name not in active,'Dependency cycle: '+name
        if name in finished:return
        active.add(name)
        for target in nodes.get(name,[]):visit(target)
        active.remove(name);finished.add(name)
    for name in starts:visit(name)


def test_legacy_exports_are_identical_neutral_objects():
    from anime_recommender import artifact_io, contracts
    from anime_recommender.training import artifacts
    from anime_recommender.preprocessing.training_data import DataValidationError
    from anime_recommender.inference.recommendations import RecommendationResult
    assert artifacts.load_models is artifact_io.load_models
    assert artifacts.save_models is artifact_io.save_models
    assert DataValidationError is contracts.DataValidationError
    assert RecommendationResult is contracts.RecommendationResult
