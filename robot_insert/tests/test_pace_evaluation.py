"""Official scoring and resumable evaluation accounting, without launching Gazebo."""
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_sfp0_pace import (evaluate, save_reports, score_run, summarize, resume_gui,
                               wait_launcher, check_policy, evaluation_format)


def run_fixture(run, *, score=75, message='Cable insertion successful.', scene=1):
    run.mkdir(parents=True)
    (run/'scoring').mkdir()
    (run/'eval.log').write_text('[INFO] [aic_engine-8]: process has finished cleanly [pid 123]\n')
    (run/'settings.json').write_text(json.dumps({'trial_ids': ['trial_1']}))
    manifest = {'checkpoint_sha256': 'frozen', 'scene_attempt': scene,
        'ground_truth': False, 'method': 'MultiViewACT', 'batch_size': 1,
        'launcher_exit_code': 0, 'session_wall_seconds': 100}
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    (run/'scoring/scoring.yaml').write_text(yaml.safe_dump({'trial_1': {
        'tier_1': {'score': 1}, 'tier_3': {'score': score, 'message': message}}}))
    return manifest


def test_partial_insertion_is_failure_not_success(tmp_path):
    run_fixture(tmp_path/'run', score=38, message='Partial insertion')
    assert score_run(tmp_path/'run', 'frozen', 1)['outcome'] == 'failure'


@pytest.mark.parametrize('change', [
    {'ground_truth': True}, {'checkpoint_sha256': 'different'}, {'scene_attempt': 2},
    {'method': 'AutoExpertACT'}, {'dry_run': True}, {'prepare_only': True},
    {'interrupted_or_timed_out': True}, {'launcher_exit_code': 1},
])
def test_assisted_wrong_scene_or_checkpoint_and_incomplete_runs_rejected(tmp_path, change):
    manifest = run_fixture(tmp_path/'run')
    manifest.update(change)
    (tmp_path/'run/run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        score_run(tmp_path/'run', 'frozen', 1)


def test_event_or_partial_score_alone_cannot_count(tmp_path):
    run_fixture(tmp_path/'run')
    (tmp_path/'run/eval.log').write_text('insertion event received')
    with pytest.raises(ValueError, match='not completed'):
        score_run(tmp_path/'run', 'frozen', 1)
    (tmp_path/'run/eval.log').write_text('[INFO] [aic_engine-8]: process has finished cleanly [pid 123]')
    (tmp_path/'run/scoring/scoring.yaml').write_text('trial_1: {}')
    with pytest.raises(ValueError, match='missing a finite'):
        score_run(tmp_path/'run', 'frozen', 1)


def test_errors_do_not_inflate_success_rate_and_pending_not_counted(tmp_path):
    rows = [{'trial': i, 'scene_attempt': i, 'outcome': outcome, 'success': outcome == 'success'}
            for i, outcome in enumerate(['success', 'failure', 'error'], 1)]
    report = save_reports(tmp_path, rows, 100)
    assert report['remaining_trials'] == 97 and not report['complete']
    assert report['success_rate_percent'] == pytest.approx(100/3)
    assert report['official_scored_success_rate_percent'] == 50
    assert summarize([], 100)['success_rate_percent'] is None
    assert len((tmp_path/'episodes.csv').read_text().splitlines()) == 4


def test_resume_recovers_finished_trial_before_result_commit_without_relaunch(tmp_path):
    run = tmp_path/'trials/001/runs/finished'
    run_fixture(run)
    contract = {'trials': 1, 'scene_start': 1, 'checkpoint_sha256': 'frozen', 'keep_scoring_bags': True}
    assert evaluate(tmp_path, contract, resume=True) == 0
    assert evaluate(tmp_path, contract, resume=True) == 0
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and report['finished_trials'] == 1
    assert report['successful_trials'] == 1 and report['success_rate_percent'] == 100
    assert json.loads((tmp_path/'trials/001/result.json').read_text())['scene_attempt'] == 1


def test_resume_100_completed_runs_counts_each_scene_once(tmp_path):
    for i in range(1, 101):
        directory = tmp_path/'trials'/f'{i:03d}'
        directory.mkdir(parents=True)
        outcome = 'success' if i <= 73 else 'failure'
        (directory/'result.json').write_text(json.dumps({'trial': i, 'scene_attempt': 41+i,
            'outcome': outcome, 'success': outcome == 'success'}))
    contract = {'trials': 100, 'scene_start': 42, 'checkpoint_sha256': 'frozen'}
    assert evaluate(tmp_path, contract, resume=True) == 0
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['successful_trials'] == 73 and report['failed_trials'] == 27
    assert report['complete'] and report['success_rate_percent'] == 73


def test_resume_rejects_changed_scene_identity(tmp_path):
    d = tmp_path/'trials/001';d.mkdir(parents=True)
    (d/'result.json').write_text(json.dumps({'trial': 1, 'scene_attempt': 99}))
    with pytest.raises(ValueError, match='identity'):
        evaluate(tmp_path, {'trials': 1, 'scene_start': 1}, resume=True)


def execution_contract():
    return {'trials': 1, 'scene_start': 1, 'checkpoint_sha256': 'frozen',
        'keep_scoring_bags': True, 'device': 'cuda', 'amp': 'bf16', 'gui': 'false',
        'renderer': 'hardware', 'real_time_factor': .5, 'max_task_seconds': 60,
        'max_wall_seconds': 600}


def test_launch_uses_frozen_weight_fixed_scene_and_no_expert(tmp_path, monkeypatch):
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            directory = Path(command[command.index('--results-dir')+1])/'run'
            run_fixture(directory)
        def wait(self, **kwargs):return 0
    monkeypatch.setattr('evaluate_sfp0_pace.subprocess.Popen', Process)
    monkeypatch.setattr('storage_checks.ensure_free_space', lambda *a, **k: None)
    assert evaluate(tmp_path, execution_contract()) == 0
    command = calls[0]
    assert '--auto-expert' not in command and '--human-control' not in command
    assert command[command.index('--checkpoint')+1] == str(tmp_path/'checkpoint.pt')
    assert command[command.index('--scene-attempt')+1] == '1'
    assert json.loads((tmp_path/'summary.json').read_text())['success_rate_percent'] == 100


def test_interrupt_keeps_trial_pending_and_stops_launcher(tmp_path, monkeypatch):
    stopped = []
    class Process:
        def __init__(self, *args, **kwargs):pass
        def wait(self, **kwargs):raise KeyboardInterrupt
    monkeypatch.setattr('evaluate_sfp0_pace.subprocess.Popen', Process)
    monkeypatch.setattr('storage_checks.ensure_free_space', lambda *a, **k: None)
    monkeypatch.setattr('evaluate_sfp0_pace.stop_launcher', stopped.append)
    assert evaluate(tmp_path, execution_contract()) == 130
    assert len(stopped) == 1
    assert not (tmp_path/'trials/001/result.json').exists()
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['finished_trials'] == 0 and report['remaining_trials'] == 1


def test_launcher_error_is_counted_and_stops_instead_of_skipping_failed_attempt(tmp_path, monkeypatch):
    class Process:
        def __init__(self, *args, **kwargs):pass
        def wait(self, **kwargs):return 2
    monkeypatch.setattr('evaluate_sfp0_pace.subprocess.Popen', Process)
    monkeypatch.setattr('storage_checks.ensure_free_space', lambda *a, **k: None)
    contract = execution_contract();contract['trials'] = 100
    assert evaluate(tmp_path, contract) == 2
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['finished_trials'] == 1 and report['error_trials'] == 1
    assert report['success_rate_percent'] == 0 and report['remaining_trials'] == 99


def test_resume_can_override_gui_without_changing_checkpoint_or_scene_contract():
    contract = {**execution_contract(), 'checkpoint_step': 3000}
    assert resume_gui(contract, None) is contract
    updated = resume_gui(contract, 'true')
    assert contract['gui'] == 'false' and updated['gui'] == 'true'
    assert updated['checkpoint_sha256'] == contract['checkpoint_sha256']
    assert updated['scene_start'] == contract['scene_start']
    assert updated['display_changes'][0]['from'] == 'false'
    assert resume_gui(updated, None)['gui'] == 'true'


def test_launcher_emits_progress_while_waiting(tmp_path, capsys):
    import subprocess
    class Process:
        def __init__(self):self.calls = 0
        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:raise subprocess.TimeoutExpired('launcher', timeout)
            return 0
    assert wait_launcher(Process(), 600, 7, tmp_path) == 0
    text = capsys.readouterr().out
    assert 'evaluation_trial_wait' in text and '"trial": 7' in text


def test_act_and_pace_checkpoints_and_evaluation_directories_cannot_be_confused():
    act = {'format': 'aic-sfp0-act-multiview-v1'}
    pace = {'format': 'aic-sfp0-act-pace-multiview-v1'}
    check_policy(act, 'act')
    check_policy(pace, 'pace')
    with pytest.raises(ValueError, match='ACT checkpoint'):
        check_policy(pace, 'act')
    with pytest.raises(ValueError, match='PACE checkpoint'):
        check_policy(act, 'pace')
    assert evaluation_format('act') != evaluation_format('pace')
    assert evaluation_format('pace') == 'aic-sfp0-pace-evaluation-v1'
    pred = {'format': 'aic-sfp0-act-pace-pred-multiview-v1'}
    check_policy(pred, 'pace_pred')
    with pytest.raises(ValueError):check_policy(pred, 'pace')
    assert evaluation_format('pace_pred') not in [evaluation_format('act'), evaluation_format('pace')]
    check_policy({'format':'aic-sfp0-flow-matching-multiview-v1'},'flow')
    check_policy({'format':'aic-sfp0-supervised-residual-flow-multiview-v1'},'residual_flow')
    with pytest.raises(ValueError):
        check_policy({'format':'aic-sfp0-supervised-residual-flow-multiview-v1'},'flow')


def test_four_view_flow_official_score_uses_the_same_success_rule(tmp_path):
    manifest=run_fixture(tmp_path/'run')
    manifest['method']='MultiViewFlow'
    (tmp_path/'run/run_manifest.json').write_text(json.dumps(manifest))
    assert score_run(tmp_path/'run','frozen',1)['outcome']=='success'
