"""Shared simulator trials must keep independent recordings, scoring and identities."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from expert_batch import make_expert_trials,import_expert_session,pending_expert_batches
from human_dataset import import_human_run,import_receipt_path
from test_human_dataset import recorded_run,profile
import pyarrow.parquet as pq


def test_scenes_are_distinct_reproducible_and_keep_single_trial_compatibility():
    source=yaml.safe_load((Path(__file__).resolve().parents[1]/'runtime/rootfs/ws_aic/install/share/aic_engine/config/sample_config.yaml').read_text())
    before=copy.deepcopy(source)
    engine,episodes=make_expert_trials(source,profile(),180,3,1234)
    again,details=make_expert_trials(source,profile(),180,3,1234)
    assert source==before and engine==again and episodes==details
    assert list(engine['trials'])==[f'episode_{i:06d}' for i in range(3)]
    assert len({tuple(e['task_board_pose'].values()) for e in episodes})==3
    for e in episodes:
        task=engine['trials'][e['trial_id']]['tasks'][e['task_id']]
        assert task['time_limit']==185 and task['port_name']=='sfp_port_0'
    single,names=make_expert_trials(source,profile(),180,1,0)
    assert list(single['trials'])==['trial_1'] and names[0]['task_id']=='task_1'


def batch_fixture(root):
    run=recorded_run(root,expert=True);raw=run/'raw'
    files=list(raw.glob('task_1*'))
    for i in range(2):
        for p in files:shutil.copyfile(p,raw/p.name.replace('task_1',f'episode_{i:06d}',1))
    # Give the second trial different actions to catch crossed recording identities.
    with sqlite3.connect(raw/'episode_000001.sqlite3') as db:
        for row_id,text in db.execute('SELECT id,metadata FROM commands').fetchall():
            record=json.loads(text);record['pose'][0]+=.01
            db.execute('UPDATE commands SET metadata=? WHERE id=?',(json.dumps(record),row_id))
    score=yaml.safe_load((run/'scoring/scoring.yaml').read_text())['trial_1']
    (run/'scoring/scoring.yaml').write_text(yaml.safe_dump({'episode_000000':score,'episode_000001':copy.deepcopy(score)}))
    episodes=[{'trial_id':f'episode_{i:06d}','task_id':f'episode_{i:06d}'} for i in range(2)]
    (run/'settings.json').write_text(json.dumps({'collection_settings':profile(),'recording_episodes':episodes}))
    (run/'run_manifest.json').write_text(json.dumps({'method':'AutoExpertACT','prepare_only':False,
        'batch_size':2,'recording_episodes':episodes,'launcher_exit_code':0,
        'source_contract':{'collection_root':str(root/'deleted-parent')}}))
    return run


def test_same_session_trials_import_separately_and_recovery_is_idempotent(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';run=batch_fixture(root)
    assert pending_expert_batches(root)==[run]
    first=import_human_run(root,run,profile(),root/'deleted-parent',status=0,expert_source='expert',
        trial_id='episode_000000',task_id='episode_000000')
    assert first['accepted'] and pending_expert_batches(root)==[run]
    entries=import_expert_session(root,run)
    assert [e['episode_index'] for e in entries]==[0,1]
    assert [e['attempt'] for e in entries]==[0,1] and pending_expert_batches(root)==[]
    for i,e in enumerate(entries):
        assert e['accepted'] and import_receipt_path(run,'expert',e['trial']).is_file()
        rows=pq.read_table(root/e['parquet'],columns=['index','episode_index','action']).to_pydict()
        assert rows['index']==list(range(i*65,(i+1)*65)) and set(rows['episode_index'])=={i}
        assert rows['action'][0][0]==pytest.approx(i*.01,abs=1e-6)
    import_expert_session(root,run)
    manifest=json.loads((root/'_collection/manifest.json').read_text())
    assert len(manifest['attempts'])==2
    # A new checkpoint trained from a batch-only collection must locate that
    # trial's calibration instead of assuming the old raw/task_1 filename.
    from act_multiview import FORMAT,FOUR_CAMERAS,index_digest
    from run_sfp0_multiview_gazebo import collection_contract
    calibration_path=run/'raw/episode_000000_calibration.json'
    calibration=json.loads(calibration_path.read_text());calibration['source_camera_info']={'side':{'k':[1.]*9}}
    calibration_path.write_text(json.dumps(calibration))
    index={'episodes':[{'episode_index':e['episode_index'],'source_attempt':e['attempt'],
        'source_sha256':e['validation']['parquet_sha256']} for e in entries]}
    index['index_content_sha256']=index_digest(index)
    (root/'index.json').write_text(json.dumps(index))
    payload={'format':FORMAT,'camera_order':list(FOUR_CAMERAS),'action_offset':1,'task':manifest['task'],
        'config':{'chunk_size':20,'dataset_root':str(root),'index_path':str(root/'index.json')},
        'data_index_sha256':index['index_content_sha256']}
    _,info,contract=collection_contract(payload)
    assert info==calibration['source_camera_info'] and contract['calibration_source']==str(calibration_path)


def test_one_failed_trial_does_not_reject_other_success_or_reuse_its_score(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';run=batch_fixture(root)
    path=run/'scoring/scoring.yaml';scores=yaml.safe_load(path.read_text())
    scores['episode_000001']['tier_3']={'score':38.,'message':'Partial insertion'}
    path.write_text(yaml.safe_dump(scores))
    entries=import_expert_session(root,run)
    assert entries[0]['accepted'] and not entries[1]['accepted']
    assert entries[1]['official_score']['tier_3']['score']==38
    assert len(list((root/'data').rglob('*.parquet')))==1
    assert (run/'raw/episode_000001.sqlite3').exists()


def test_failed_session_cannot_be_imported_as_successful_batch(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';run=batch_fixture(root)
    entries=import_expert_session(root,run,status=1)
    assert not any(e['accepted'] for e in entries)
    assert not list((root/'data').rglob('*.parquet'))


def test_batch_monitor_waits_for_every_official_trial(tmp_path):
    from sideview_engine_done import completed
    (tmp_path/'scoring').mkdir()
    (tmp_path/'eval.log').write_text('[INFO] [aic_engine-8]: process has finished cleanly [pid 123]')
    (tmp_path/'settings.json').write_text(json.dumps({'trial_ids':['episode_000000','episode_000001']}))
    path=tmp_path/'scoring/scoring.yaml';path.write_text('episode_000000: {}\n')
    assert not completed(tmp_path)
    path.write_text('episode_000000: {}\nepisode_000001: {}\n')
    assert completed(tmp_path)


def test_only_completed_official_scoring_bags_are_removed(tmp_path):
    from expert_batch import prune_scoring_bags
    (tmp_path/'scoring/bag_episode_000000').mkdir(parents=True)
    bag=tmp_path/'scoring/bag_episode_000000/data.mcap';bag.write_bytes(b'large scoring bag')
    (tmp_path/'raw').mkdir();raw=tmp_path/'raw/episode_000000.images';raw.write_bytes(b'keep original camera pack')
    (tmp_path/'eval.log').write_text('[INFO] engine still executing')
    (tmp_path/'settings.json').write_text(json.dumps({'trial_ids':['episode_000000']}))
    (tmp_path/'scoring/scoring.yaml').write_text('episode_000000: {}\n')
    with pytest.raises(ValueError,match='before all official'):prune_scoring_bags(tmp_path)
    assert bag.exists()
    (tmp_path/'eval.log').write_text('[INFO] [aic_engine-8]: process has finished cleanly [pid 123]')
    report=prune_scoring_bags(tmp_path)
    assert not bag.exists() and report['reclaimed_bytes']==len(b'large scoring bag')
    assert (tmp_path/'scoring/scoring.yaml').is_file() and raw.read_bytes()==b'keep original camera pack'


def test_recover_completed_batch_if_launcher_died_before_saving_exit_status(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';run=batch_fixture(root)
    path=run/'run_manifest.json';manifest=json.loads(path.read_text());manifest.pop('launcher_exit_code')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='no recorded exit status'):import_expert_session(root,run)
    settings_path=run/'settings.json';settings=json.loads(settings_path.read_text())
    settings['trial_ids']=[e['trial_id'] for e in settings['recording_episodes']]
    settings_path.write_text(json.dumps(settings))
    (run/'eval.log').write_text('[INFO] [aic_engine-8]: process has finished cleanly [pid 123]')
    assert all(e['accepted'] for e in import_expert_session(root,run))


def test_first_stop_keeps_batch_running_second_stop_only_signals_its_process_group(monkeypatch):
    import collect_sfp0_expert as collector
    child=NS(pid=12345,wait=Mock(side_effect=[KeyboardInterrupt(),0]),poll=Mock(return_value=None))
    launch=Mock(return_value=child);kill=Mock()
    monkeypatch.setattr(collector.subprocess,'Popen',launch);monkeypatch.setattr(collector.os,'killpg',kill)
    assert collector.run_collection_session(['collect-child'])==(0,True)
    assert launch.call_args.kwargs['start_new_session'] and not kill.called
    child.wait=Mock(side_effect=[KeyboardInterrupt(),KeyboardInterrupt(),130])
    assert collector.run_collection_session(['collect-child'])==(130,True)
    kill.assert_called_once_with(12345,collector.signal.SIGINT)
