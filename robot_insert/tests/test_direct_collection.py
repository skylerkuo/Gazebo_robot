"""Full controller trajectories append without losing assisted data/provenance."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_sfp0_direct import import_direct_run,pending_direct_runs
from human_dataset import import_human_run
from sideview_data import TASK
from test_human_dataset import recorded_run,profile


def direct_fixture(root,*,mixed=False,failed=False):
    run=recorded_run(root,expert=True)
    summary_path=run/'raw/task_1.json'
    summary=json.loads(summary_path.read_text())
    summary['controller']='official_cheatcode_only'
    summary_path.write_text(json.dumps(summary))
    with sqlite3.connect(run/'raw/task_1.sqlite3') as db:
        for row_id,text in db.execute('SELECT id,metadata FROM commands').fetchall():
            record=json.loads(text)
            record.update(control_source='expert',control_segment=0,human_active=True)
            if mixed and row_id==1:record['control_source']='model'
            db.execute('UPDATE commands SET metadata=? WHERE id=?',(json.dumps(record),row_id))
    for path in list((run/'raw').glob('task_1*')):
        path.rename(path.with_name(path.name.replace('task_1','trial_1',1)))
    (run/'settings.json').write_text(json.dumps(dict(profile(),minimum_frames=21,
        recording_mode='official_cheatcode_only',trial_ids=['trial_1'])))
    (run/'launcher.json').write_text(json.dumps({'exit_code':1 if failed else 0}))
    return run


def existing_assisted(root):
    run=recorded_run(root,expert=True)
    # Keep its immutable data/receipt alongside a differently named direct session.
    old=run.with_name('assisted_session');run.rename(old)
    entry=import_human_run(root,old,profile(),root/'parent',status=0,expert_source='expert')
    assert entry['accepted']
    return entry


def test_direct_data_appends_all_expert_four_view_trajectories_and_recovers_idempotently(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';old=existing_assisted(root)
    old_bytes=(root/old['parquet']).read_bytes()
    run=direct_fixture(root)
    assert pending_direct_runs(root)==[run]
    entries=import_direct_run(root,run)
    entry=entries[0]
    assert entry['accepted'] and entry['episode_index']==1
    assert entry['outcome']=='expert_direct_success' and entry['recording_type']=='official_cheatcode_only'
    assert entry['validation']['control_source_counts']=={'expert':65}
    assert entry['validation_eligible'] and len(entry['expert_validation_frames'])==45
    rows=pq.read_table(root/entry['parquet']).to_pydict()
    assert set(rows['action_source'])=={'expert'} and all(rows['human_active'])
    assert rows['index']==list(range(65,130))
    for camera in ('left','center','right','side'):
        assert len(rows[f'observation.images.{camera}_camera'])==65
    assert pending_direct_runs(root)==[]
    assert import_direct_run(root,run)==entries
    assert len(json.loads((root/'_collection/manifest.json').read_text())['attempts'])==2
    assert (root/old['parquet']).read_bytes()==old_bytes


@pytest.mark.parametrize('problem',['mixed','failed','no_receipt'])
def test_unconfirmed_or_mixed_direct_rollouts_never_become_direct_demonstrations(tmp_path,monkeypatch,problem):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';old=existing_assisted(root)
    run=direct_fixture(root,mixed=problem=='mixed',failed=problem=='failed')
    if problem=='no_receipt':
        (run/'launcher.json').unlink()
        with pytest.raises(FileNotFoundError):import_direct_run(root,run)
    else:
        assert not import_direct_run(root,run)[0]['accepted']
    assert len(list((root/'data').rglob('*.parquet')))==1
    assert (root/old['parquet']).exists()


@pytest.mark.parametrize('direct',[False,True])
def test_policy_uses_official_teacher_and_marks_actual_command_source(monkeypatch,direct):
    class Teacher:
        def insert_cable(self,task,get_observation,move_robot,send_feedback):
            move_robot(motion_update='accepted_command')
            return True
    teacher=ModuleType('aic_example_policies.ros.CheatCode');teacher.CheatCode=Teacher
    recorder_module=ModuleType('sideview_recorder');recorder_module.Recorder=Mock()
    monkeypatch.setitem(sys.modules,'aic_example_policies.ros.CheatCode',teacher)
    monkeypatch.setitem(sys.modules,'sideview_recorder',recorder_module)
    path=Path(__file__).resolve().parents[1]/'policies/CollectSFP.py'
    spec=importlib.util.spec_from_file_location('direct_teacher_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    policy=object.__new__(module.CollectSFP)
    policy.settings={'recording_mode':'official_cheatcode_only'} if direct else {}
    policy.recorder=Mock()
    task=NS(target_module_name=TASK['target_module'],plug_name=TASK['plug'],port_name=TASK['port'])
    move=Mock(return_value=True)
    assert policy.insert_cable(task,None,move,None)
    move.assert_called_once_with(motion_update='accepted_command',joint_motion_update=None)
    if direct:
        policy.recorder.command.assert_called_once_with('accepted_command',source='expert',segment=0,human_active=True)
        policy.recorder.finish.assert_called_once_with(True,controller='official_cheatcode_only')
    else:
        policy.recorder.command.assert_called_once_with('accepted_command')
        policy.recorder.finish.assert_called_once_with(True,controller='teacher')
