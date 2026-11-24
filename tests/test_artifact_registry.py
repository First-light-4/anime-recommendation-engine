"""No-fit synthetic bundle, immutable publication and extensible registry checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from implicit.cpu.als import AlternatingLeastSquares

from anime_recommender import artifact_io
from anime_recommender.artifact_files import atomic_json, file_sha256
from anime_recommender.contracts import DataValidationError
from anime_recommender.inference.bundle import ModelBundle
from anime_recommender.registry.models import ModelRegistry


@pytest.fixture
def bundle(monkeypatch):
    config = {"factors": 2, "regularization": .1, "iterations": 1, "random_state": 42,
        "num_threads": 1, "alpha": 1., "dtype": "float32", "use_cg": True, "cg_steps": 3}
    als = AlternatingLeastSquares(**{k:v for k,v in config.items() if k != "cg_steps"})
    als.cg_steps = 3
    als.item_factors = np.array([[1,0],[0,1],[.5,.5]],dtype="float32")
    als.user_factors = np.array([[.8,.2],[.2,.8]],dtype="float32")
    items = pd.DataFrame({"dataset_anime_id":[1,2,3],"myanimelist_id":[101,102,103],"matrix_col":[0,1,2]})
    users = pd.DataFrame({"dataset_user_id":[11,12],"matrix_row":[0,1]})
    popularity = items.assign(unique_users=[2,1,1])
    monkeypatch.setattr(AlternatingLeastSquares,"fit",lambda *a,**k:pytest.fail("No fit"))
    monkeypatch.setattr("requests.sessions.Session.request",lambda *a,**k:pytest.fail("No network"))
    return ModelBundle(als,popularity,items,users,{"format_version":1,"training_config":{"als":config},
        "dataset_config":{"positive_threshold":5},"train_fingerprint":"synthetic-train"})


def test_publish_load_and_repeat_never_overwrites(bundle,tmp_path):
    path=tmp_path/"bundle"
    original=bundle.recommenders()["als"].recommend([1],2).recommendations
    artifact_io.save_models(bundle,path)
    hashes={p.name:file_sha256(p) for p in path.iterdir()}
    loaded=artifact_io.load_models(path,bundle.items,"synthetic-train")
    pd.testing.assert_frame_equal(original,loaded.recommenders()["als"].recommend([1],2).recommendations)
    with pytest.raises(FileExistsError,match="already exists"):
        artifact_io.save_models(bundle,path)
    assert hashes=={p.name:file_sha256(p) for p in path.iterdir()}
    assert not list(tmp_path.glob(".*publish.lock"))


@pytest.mark.parametrize("failure",[OSError,KeyboardInterrupt])
def test_interrupted_write_has_no_published_bundle_or_metadata_mutation(bundle,tmp_path,monkeypatch,failure):
    before=deepcopy(bundle.metadata)
    def stop(*args,**kwargs):raise failure("simulated write interruption")
    monkeypatch.setattr(artifact_io,"atomic_parquet",stop)
    with pytest.raises(failure):artifact_io.save_models(bundle,tmp_path/"bundle")
    assert not (tmp_path/"bundle").exists()
    assert not list(tmp_path.iterdir())
    assert bundle.metadata==before


def test_destination_appearing_at_publish_is_not_replaced(bundle,tmp_path,monkeypatch):
    publish=artifact_io._publish_directory
    def race(staging,destination):
        destination.mkdir()
        publish(staging,destination)
    monkeypatch.setattr(artifact_io,"_publish_directory",race)
    with pytest.raises(FileExistsError):artifact_io.save_models(bundle,tmp_path/"bundle")
    assert (tmp_path/"bundle").is_dir() and not list((tmp_path/"bundle").iterdir())
    assert list(tmp_path.iterdir())==[tmp_path/"bundle"]


def test_stale_lock_is_explicit_and_not_auto_removed(bundle,tmp_path):
    lock=tmp_path/".bundle.publish.lock"
    lock.write_text("12345")
    with pytest.raises(FileExistsError,match="locked"):
        artifact_io.save_models(bundle,tmp_path/"bundle")
    assert lock.read_text()=="12345"


def test_schema_incomplete_corruption_and_fingerprints(bundle,tmp_path):
    path=tmp_path/"bundle"
    artifact_io.save_models(bundle,path)
    with pytest.raises(DataValidationError,match="different training"):
        artifact_io.load_models(path,expected_train_fingerprint="wrong")
    wrong=bundle.items.copy();wrong["myanimelist_id"]+=1
    with pytest.raises(DataValidationError,match="different model"):
        artifact_io.load_models(path,expected_mapping=wrong)
    metadata=json.loads((path/"metadata.json").read_text())
    atomic_json(path/"metadata.json",{**metadata,"format_version":999})
    with pytest.raises(DataValidationError,match="format_version"):artifact_io.load_models(path)
    atomic_json(path/"metadata.json",metadata)
    (path/"popularity.parquet").write_bytes(b"corrupt")
    with pytest.raises(DataValidationError,match="checksum"):artifact_io.load_models(path)
    # Even a matching manifest cannot make a malformed native payload loadable.
    metadata["files_sha256"]["popularity.parquet"]=file_sha256(path/"popularity.parquet")
    atomic_json(path/"metadata.json",metadata)
    with pytest.raises(DataValidationError,match="Invalid model bundle"):artifact_io.load_models(path)
    (path/"popularity.parquet").unlink()
    with pytest.raises(FileNotFoundError,match="incomplete"):artifact_io.load_models(path)


def test_registry_versions_ids_legacy_metadata_and_config_fingerprints(bundle,tmp_path):
    (tmp_path/"configs").mkdir()
    entries={}
    for version in ("1k","5k","10k","20k"):
        path=tmp_path/"artifacts/models"/version
        bundle.metadata.pop("model_version",None)
        if version!="1k":bundle.metadata["model_version"]=version
        artifact_io.save_models(bundle,path)
        entries[version]={"model_dir":str(path.relative_to(tmp_path)),"model_id":"als-"+version}
    config={"schema_version":1,"default_version":"10k","versions":entries}
    atomic_json(tmp_path/"configs/model_versions.json",config)
    registry=ModelRegistry(tmp_path)
    assert registry.default_version=="10k"
    assert len(registry.list_models())==4
    assert registry.load().metadata["model_version"]=="10k"
    assert registry.load("als-20k").metadata["model_version"]=="20k"
    assert "model_version" not in registry.load("1k").metadata
    assert registry.verify("20k").identity.model_id=="als-20k"
    entries["20k"]["mapping_fingerprint"]="wrong"
    atomic_json(tmp_path/"configs/model_versions.json",config)
    with pytest.raises(DataValidationError,match="mapping fingerprint"):
        ModelRegistry(tmp_path).load("20k")
    entries["missing"]={"model_dir":"artifacts/models/missing"}
    atomic_json(tmp_path/"configs/model_versions.json",config)
    registry=ModelRegistry(tmp_path)
    assert not registry.is_available("missing")
    with pytest.raises(DataValidationError,match="missing"):registry.load("missing")


def test_two_processes_publish_exactly_once(bundle,tmp_path):
    source,target=tmp_path/"source",tmp_path/"target"
    artifact_io.save_models(bundle,source)
    code='''import sys
from anime_recommender.artifact_io import load_models,save_models
bundle=load_models(sys.argv[1])
try: save_models(bundle,sys.argv[2])
except FileExistsError: sys.exit(10)
'''
    options={"stdout":subprocess.PIPE,"stderr":subprocess.PIPE,"text":True}
    if os.name=="nt":options["creationflags"]=subprocess.CREATE_NO_WINDOW
    processes=[subprocess.Popen([sys.executable,"-c",code,str(source),str(target)],**options) for _ in range(2)]
    results=[(p.communicate(timeout=40),p.returncode) for p in processes]
    assert sorted(code for _,code in results)==[0,10],results
    assert artifact_io.load_models(target).metadata["mapping_fingerprint"]==bundle.metadata["mapping_fingerprint"]
    assert not list(tmp_path.glob(".*publish.lock"))
