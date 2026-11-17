"""Bounded acquisition with fake clocks/APIs; never external network or model fits."""

import copy
from datetime import datetime, timezone
from email.utils import format_datetime
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")

from anime_recommender.ingestion.metadata_requests import MetadataRequests, RunStopped, retry_after_seconds
from anime_recommender.ingestion.viewing_order_metadata import run_metadata_pipeline
from anime_recommender.ingestion.franchise_artifacts import load_prepared_mapping
from anime_recommender.ingestion.yummyanime import YummyAnimeError, YummyAnimeAuthorizationError
from anime_recommender.preprocessing.metadata_orders import merge_metadata_orders
from anime_recommender.preprocessing.franchise_mapping import build_franchise_mapping, load_franchise_resources
from anime_recommender.ingestion.training_data import atomic_json


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self): self.now, self.sleeps = 10000.0, []
    def __call__(self): return self.now
    def sleep(self, seconds): self.sleeps.append(seconds); self.now += seconds


def order(ids=(11, 12, 13, 14), group=7):
    return [{"anime_id": value, "type": {"alias": "tv"}, "data": {"id": group, "index": i,
        "text": "адаптация манги" if i == 0 else "продолжение"}, "user": {"private": "discard"}}
        for i, value in enumerate(ids)]


@pytest.fixture
def setup(tmp_path):
    config = json.loads((ROOT / "configs/viewing_order_metadata.json").read_text())
    config.update(bootstrap_path=None, supplement_path=None)
    source = tmp_path / config["vocabulary_path"]; source.parent.mkdir(parents=True)
    pd.DataFrame({"dataset_anime_id": [1, 2, 3], "myanimelist_id": [101, 102, 103], "matrix_col": [0, 1, 2]}).to_parquet(source,index=False)
    client = Mock()
    client.get_anime_by_mal_ids.side_effect = lambda ids: [{"anime_id": value-90, "remote_ids": {"myanimelist_id": value}, "user": {"private": "discard"}} for value in ids]
    client.get_anime.return_value = {"anime_id": 11, "viewing_order": order(), "user": {"private": "discard"}}
    client.get_anime_by_ids.side_effect = lambda ids: [{"anime_id": value, "remote_ids": {"myanimelist_id": value+90}} for value in ids]
    return tmp_path, config, client, Clock()


def run(setup, **kwargs):
    root, config, client, clock = setup
    return run_metadata_pipeline(root, config, client=client, clock=clock, sleep=clock.sleep, **kwargs)


def test_dry_run_no_network_no_write_and_large_run_guard(setup):
    root, config, client, _ = setup
    report = run(setup)
    assert report["dry_run"] and report["requests"] == 0 and report["train_items"] == 3
    assert not (root / config["output_dir"]).exists()
    assert client.mock_calls == []
    config["max_requests"] = 11
    with pytest.raises(ValueError, match="authorization"): run(setup, execute=True)
    assert client.mock_calls == []


def test_discovery_group_dedup_external_members_incremental_and_fingerprint(setup):
    root, config, client, clock = setup
    first = run(setup, execute=True)
    assert first["requests"] == 3  # One MAL batch, one group detail, one external ID batch.
    assert first["known_franchise_groups"] == 1 and first["mapped_parts"] == 4
    assert first["covered_train_items"] == 3 and first["train_coverage"] == 1
    client.get_anime.assert_called_once_with(11)
    snapshot, _ = load_prepared_mapping(root / config["output_dir"])
    assert {n["mal_id"] for n in snapshot["nodes"]} == {101,102,103,104}
    calls = client.mock_calls.copy()
    second = run(setup, execute=True)
    assert second["requests"] == 0 and client.mock_calls == calls
    assert second["mapping_fingerprint"] == first["mapping_fingerprint"]
    for path in (root / config["output_dir"]).rglob('*.json'):
        assert 'discard' not in path.read_text() and 'private' not in path.read_text()
    # Adding a train item discovers it without refetching the old known group.
    pd.DataFrame({"dataset_anime_id":[1,2,3,5],"myanimelist_id":[101,102,103,105],"matrix_col":[0,1,2,3]}).to_parquet(root/config["vocabulary_path"],index=False)
    client.get_anime.return_value = {"anime_id":15,"viewing_order":order((11,12,13,14,15))}
    third = run(setup, execute=True)
    assert third["requests"] == 2 and third["mapped_parts"] == 5
    assert client.get_anime_by_mal_ids.call_args.args[0] == [105]
    assert client.get_anime.call_args.args[0] == 15
    assert third["known_franchise_groups"] == 1


def test_budget_stop_resume_and_interruption(setup):
    root, config, client, clock = setup
    config["max_requests"] = 1
    first = run(setup, execute=True)
    assert first["stop_reason"] == "request_budget" and first["requests"] == 1
    assert first["unmapped_parts"] == 0
    config["max_requests"] = 5
    client.get_anime.side_effect = KeyboardInterrupt
    interrupted = run(setup, execute=True)
    assert interrupted["stop_reason"] == "interrupted"
    client.get_anime.side_effect = None
    resumed = run(setup, execute=True)
    assert resumed["known_franchise_groups"] == 1
    assert client.get_anime_by_mal_ids.call_count == 1
    assert client.get_anime_by_ids.call_count == 1


def test_rate_limit_retry_after_and_temporary_backoff(setup):
    _, config, _, clock = setup
    state = {}; checkpoints = Mock()
    api = MetadataRequests(config, state, checkpoints, clock, clock.sleep)
    action = Mock(side_effect=[YummyAnimeError('temporary',429,retry_after='5'), []])
    assert api.call('catalog',action) == []
    assert action.call_count == 2 and clock.sleeps == [5]
    api.call('next',lambda: [])
    assert clock.sleeps[-1] == 2  # Request-rate floor still applies.
    other = MetadataRequests(config, {}, Mock(), clock, clock.sleep)
    operation = Mock(side_effect=[YummyAnimeError('temp',503),YummyAnimeError('temp',503),[]])
    assert other.call('detail',operation) == []
    assert clock.sleeps[-2:] == [2,4]
    date = format_datetime(datetime.fromtimestamp(clock()+10,timezone.utc),usegmt=True)
    assert retry_after_seconds(date,clock()) == 10


def test_long_retry_after_is_deferred_and_permanent_auth_stops(setup):
    root, config, client, clock = setup
    client.get_anime_by_mal_ids.side_effect = YummyAnimeError('rate',429,retry_after='120')
    deferred = run(setup, execute=True)
    assert deferred["stop_reason"] == "deferred_rate_limit" and deferred["requests"] == 1
    assert clock.sleeps == []
    again = run(setup, execute=True)
    assert again["requests"] == 0 and again["stop_reason"] == "deferred_rate_limit"
    assert client.get_anime_by_mal_ids.call_count == 1
    clock.now += 120
    client.get_anime_by_mal_ids.side_effect = YummyAnimeAuthorizationError('HTTP 403',403)
    denied = run(setup, execute=True)
    assert denied["stop_reason"] == "authorization_error" and denied["requests"] == 1
    client.get_anime.assert_not_called()


def test_negative_cache_and_refresh(setup):
    _, config, client, clock = setup
    client.get_anime_by_mal_ids.side_effect = None
    client.get_anime_by_mal_ids.return_value = []
    first = run(setup,execute=True)
    assert first["selected_status_counts"]["missing"] == 3
    assert run(setup,execute=True)["requests"] == 0
    clock.now += config["negative_refresh_days"]*86400+1
    assert run(setup,execute=True)["requests"] == 1
    client.get_anime.assert_not_called()


def test_refresh_retains_old_safe_data_on_temporary_error(setup):
    _, config, client, clock = setup
    first = run(setup,execute=True)
    clock.now += config["refresh_days"]*86400+1
    client.get_anime.side_effect = YummyAnimeError('temporary',503)
    # MAL batch and detail retry attempts fit the five-request budget.
    refreshed = run(setup,execute=True)
    assert refreshed["mapped_parts"] == first["mapped_parts"]
    assert refreshed["stale_orders"] >= 1


def test_overlap_conflicts_gaps_alternatives_and_unmapped_barrier():
    a={"seed_yummy_id":11,"viewing_order":order((11,12),group=7)}
    b={"seed_yummy_id":12,"viewing_order":order((12,13),group=8)}
    b['viewing_order'][0]['data']['text']='продолжение'
    resolved=[{"anime_id":y,"myanimelist_id":y+90} for y in [11,12,13]]
    merged=merge_metadata_orders([a,b],resolved)
    mapping=build_franchise_mapping(merged)
    assert len(set(mapping.franchise.values()))==1
    assert mapping.viewing_path(103)==((101,),(102,),(103,))
    reversed_b=copy.deepcopy(a); reversed_b['seed_yummy_id']=12
    reversed_b['viewing_order']=list(reversed(reversed_b['viewing_order']))
    for i,row in enumerate(reversed_b['viewing_order']):row['data']['index']=i
    conflict=merge_metadata_orders([a,reversed_b],resolved)
    assert conflict['merge_quality'][0]['conflicting']
    assert not conflict['ordered_sequences'][0]['order_known']
    missing=merge_metadata_orders([a,b],[resolved[0],resolved[2]])
    assert missing['ordered_sequences'][0]['steps'][1]['mal_ids']==[]
    gap=copy.deepcopy(a);gap['viewing_order'][1]['data']['index']=2
    assert not merge_metadata_orders([gap],resolved)['ordered_sequences'][0]['order_known']
    alternative=copy.deepcopy(a);alternative['viewing_order'][1]['data']['text']='альтернативная версия'
    alt=merge_metadata_orders([alternative],resolved)
    assert alt['ordered_sequences'][0]['steps']==[{'mal_ids':[101],'yummy_ids':[11],'source_index':0}]
    assert 102 in alt['ordered_sequences'][0]['optional_mal_ids']  # Not invented mandatory continuation.


def test_conflicting_identifiers_and_local_loader_fallback(setup):
    root, config, client, _ = setup
    client.get_anime_by_mal_ids.side_effect=None
    client.get_anime_by_mal_ids.return_value=[{"anime_id":11,"remote_ids":{"myanimelist_id":101}},
        {"anime_id":99,"remote_ids":{"myanimelist_id":101}}]
    result=run(setup,execute=True)
    assert result['ambiguous_identifiers']>=1
    client.get_anime.assert_not_called()
    cfg_dir=root/'configs';cfg_dir.mkdir()
    atomic_json(cfg_dir/'franchise_filter.json',{'enabled':True,'relations_path':'configs/missing.json',
        'prepared_dir':config['output_dir'],'max_recommendations_per_franchise':1,'candidate_multiplier':3})
    cfg,mapping=load_franchise_resources(root)
    assert not mapping.franchise and cfg['metadata_warning']
    latest=json.loads((root/config['output_dir']/'latest.json').read_text())
    (root/config['output_dir']/latest['version']/'mapping.json').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):load_prepared_mapping(root/config['output_dir'])
    cfg,mapping=load_franchise_resources(root)
    assert not mapping.franchise and 'No valid' in cfg['metadata_warning']


def test_rascal_pair_discovered_and_new_user_top15_has_one_part(setup):
    from anime_recommender.inference.franchise_filter import filter_candidates
    root,config,client,_=setup
    pd.DataFrame({'dataset_anime_id':[1,2],'myanimelist_id':[37450,38329],'matrix_col':[0,1]}).to_parquet(root/config['vocabulary_path'],index=False)
    client.get_anime_by_mal_ids.side_effect=None
    client.get_anime_by_mal_ids.return_value=[{'anime_id':111,'remote_ids':{'myanimelist_id':37450}},
        {'anime_id':112,'remote_ids':{'myanimelist_id':38329}}]
    rows=order((111,112))
    rows[1]['type']['alias']='movie'  # Explicit continuation, not mandatory merely by format.
    client.get_anime.return_value={'anime_id':111,'viewing_order':rows}
    report=run(setup,execute=True)
    assert report['requests']==2 and report['known_franchise_groups']==1
    client.get_anime.assert_called_once_with(111)
    snapshot,_=load_prepared_mapping(root/config['output_dir']);mapping=build_franchise_mapping(snapshot)
    mals=[38329,37450]+list(range(900001,900015))
    candidates=pd.DataFrame({'dataset_anime_id':range(1,17),'myanimelist_id':mals,
        'score':[1-i/100 for i in range(16)],'rank':range(1,17),'model_name':['als']*16})
    result=filter_candidates(candidates,mapping,top_k=15)
    assert len(result.recommendations)==15
    assert 37450 in set(result.recommendations.myanimelist_id)
    assert 38329 not in set(result.recommendations.myanimelist_id)
    # This is explicitly a synthetic API regression, not a live source claim.


def test_confirmed_alternative_stage_artifact_and_no_title_inference(setup):
    from anime_recommender.inference.franchise_filter import filter_candidates
    root,config,client,_=setup
    cfg=root/'configs';cfg.mkdir()
    supplement={'schema_version':1,'source':'synthetic verified structured relation',
        'relations':[{'source_mal_id':101,'target_mal_id':102,'relation':'ALTERNATIVE'}],
        'reviewed_policy':{'mainline_mal_ids':[101,102]}}
    atomic_json(cfg/'supplement.json',supplement);config['supplement_path']='configs/supplement.json'
    rows=order((11,12,13));rows[1]['data']['text']='альтернативная версия'
    client.get_anime.return_value={'anime_id':11,'viewing_order':rows}
    report=run(setup,execute=True)
    snapshot,_=load_prepared_mapping(root/config['output_dir']);mapping=build_franchise_mapping(snapshot)
    assert mapping.sequence_state(101,[102])['next_mal_ids']==[103]
    latest=json.loads((root/config['output_dir']/'latest.json').read_text())
    parts=pd.read_parquet(root/config['output_dir']/latest['version']/'parts.parquet').set_index('myanimelist_id')
    assert parts.loc[102,'part_role']=='alternative'
    assert parts.loc[102,'stage_group']==0


def test_no_local_metadata_uses_exact_seen_filter_and_warning(setup):
    from anime_recommender.inference.franchise_filter import filter_candidates
    root,config,client,_=setup
    cfg=root/'configs';cfg.mkdir()
    atomic_json(cfg/'franchise_filter.json',{'enabled':True,'relations_path':'configs/missing.json',
        'prepared_dir':config['output_dir'],'max_recommendations_per_franchise':1,'candidate_multiplier':3})
    loaded,mapping=load_franchise_resources(root)
    assert 'No valid local' in loaded['metadata_warning']
    candidates=pd.DataFrame({'dataset_anime_id':[1,2],'myanimelist_id':[101,102],
        'score':[.9,.8],'rank':[1,2],'model_name':['als','als']})
    result=filter_candidates(candidates,mapping,observed_mal_ids=[101])
    assert result.recommendations.myanimelist_id.tolist()==[102]
    assert result.diagnostics['covered_candidate_share']==0
    assert client.mock_calls==[]


def test_detail_identity_mismatch_is_not_published_as_a_franchise(setup):
    root,config,client,_=setup
    client.get_anime.return_value={'anime_id':9999,'viewing_order':order()}
    report=run(setup,execute=True)
    assert report['known_franchise_groups']==0
    assert report['errors_total']>0
    snapshot,_=load_prepared_mapping(root/config['output_dir'])
    assert snapshot['nodes']==[]


def test_fresh_complete_refresh_supersedes_stale_duplicate_page(setup):
    root,config,client,clock=setup
    run(setup,execute=True)
    # Add an overlapping source page, then let both source pages become stale.
    state_path=root/config['output_dir']/'state.json'
    state=json.loads(state_path.read_text())
    state['orders']['12']={'seed_yummy_id':12,'viewing_order':order(),
        'status':'ordered','fetched_at':clock()}
    atomic_json(state_path,state)
    clock.now+=config['refresh_days']*86400+1
    client.get_anime.return_value={'anime_id':11,'viewing_order':order((11,13,12,14))}
    report=run(setup,execute=True)
    assert report['conflicting_groups']==0
    state=json.loads(state_path.read_text())
    assert state['orders']['12']['superseded_by']==11


def test_502_checkpoint_is_retried_after_recovery_without_negative_ttl_wait(setup):
    root,config,client,clock=setup
    config['max_retries']=0
    successful=client.get_anime_by_mal_ids.side_effect
    client.get_anime_by_mal_ids.side_effect=YummyAnimeError('HTTP 502',502)
    failed=run(setup,execute=True)
    assert failed['requests']==1
    state=json.loads((root/config['output_dir']/'state.json').read_text())
    assert all(state['mal'][str(m)]['status']=='error' and state['mal'][str(m)]['retryable'] for m in [101,102,103])
    client.get_anime_by_mal_ids.side_effect=successful
    recovered=run(setup,execute=True)
    assert recovered['selected_status_counts']['mapped']==3
    assert recovered['known_franchise_groups']==1
    assert recovered['requests']==3
    assert clock()<10000+config['negative_refresh_days']*86400
    assert client.get_anime_by_mal_ids.call_count==2
