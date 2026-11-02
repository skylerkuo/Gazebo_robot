#!/usr/bin/env python3
"""Evaluate a frozen four-view policy on reproducible Gazebo scenes (PACE entry)."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

from sideview_data import digest_file, json_write, successful_score
from sideview_engine_done import completed

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'aic-sfp0-pace-evaluation-v1'
POLICY_FORMATS = {'act': 'aic-sfp0-act-multiview-v1',
                  'pace': 'aic-sfp0-act-pace-multiview-v1',
                  'pace_pred': 'aic-sfp0-act-pace-pred-multiview-v1',
                  'flow': 'aic-sfp0-flow-matching-multiview-v1',
                  'residual_flow': 'aic-sfp0-supervised-residual-flow-multiview-v1',
                  'stacked_residual_flow': 'aic-sfp0-stacked-residual-flow-multiview-v1',
                  'alternating_residual_flow': 'aic-sfp0-alternating-act-residual-flow-multiview-v1'}


def check_policy(payload, policy_type):
    if payload.get('format') != POLICY_FORMATS[policy_type]:
        raise ValueError(f'Select a four-view {policy_type.upper()} checkpoint')


def evaluation_format(policy_type):
    return f'aic-sfp0-{policy_type}-evaluation-v1'


def summarize(records, requested):
    successes = sum(r['outcome'] == 'success' for r in records)
    failures = sum(r['outcome'] == 'failure' for r in records)
    errors = sum(r['outcome'] == 'error' for r in records)
    attempted = len(records)
    scored = successes + failures
    return {'requested_trials': requested, 'finished_trials': attempted,
        'successful_trials': successes, 'failed_trials': failures, 'error_trials': errors,
        'remaining_trials': requested-attempted, 'complete': attempted == requested,
        'success_rate_percent': 100*successes/attempted if attempted else None,
        'official_scored_trials': scored,
        'official_scored_success_rate_percent': 100*successes/scored if scored else None,
        'rate_definition': 'successes / all finished attempts, including infrastructure errors; interrupted trials are pending'}


def save_reports(output, records, requested):
    report = summarize(records, requested)
    json_write(output/'summary.json', report)
    temporary = output/'episodes.csv.partial'
    fields = ['trial', 'scene_attempt', 'outcome', 'success', 'tier_1_score',
              'tier_3_score', 'wall_seconds', 'run', 'error']
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output/'episodes.csv')
    print(json.dumps({'type': 'evaluation_progress', **report}), flush=True)
    return report


def score_run(run, checkpoint_sha256, scene_attempt):
    """A partial score or insertion event alone never qualifies as success."""
    manifest = json.loads((run/'run_manifest.json').read_text())
    if (manifest.get('checkpoint_sha256') != checkpoint_sha256
            or manifest.get('scene_attempt') != scene_attempt
            or manifest.get('ground_truth') is not False
            or manifest.get('method') not in ('MultiViewACT','MultiViewFlow')
            or manifest.get('dry_run') or manifest.get('prepare_only')
            or manifest.get('batch_size') != 1
            or manifest.get('interrupted_or_timed_out', False)):
        raise ValueError('Run does not match the frozen autonomous evaluation contract')
    if manifest.get('launcher_exit_code') not in (None, 0):
        raise ValueError('Gazebo launcher exited with an error')
    if not completed(run):
        raise ValueError('Official engine has not completed all scores')
    scores = yaml.safe_load((run/'scoring/scoring.yaml').read_text())
    score = scores['trial_1']
    # An empty score dictionary is not a valid scored policy failure.
    for tier in ('tier_1', 'tier_3'):
        value = score.get(tier, {}).get('score')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Official result is missing a finite {tier} score')
    success = successful_score(score)
    return {'outcome': 'success' if success else 'failure', 'success': success,
        'tier_1_score': score['tier_1']['score'], 'tier_3_score': score['tier_3']['score'],
        'official_score': score, 'wall_seconds': manifest.get('session_wall_seconds'),
        'run': str(run), 'error': None}


def latest_run(trial_dir):
    paths = list((trial_dir/'runs').glob('*/run_manifest.json'))
    return max(paths, key=lambda p: p.stat().st_mtime_ns).parent if paths else None


def stop_launcher(process):
    # Let the existing launcher tear down only its own simulator session.
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)


def resume_gui(contract, gui):
    """Display mode may change without changing the frozen evaluation policy."""
    if gui is None or gui == contract['gui']:
        return contract
    updated = dict(contract)
    updated['display_changes'] = [*contract.get('display_changes', []), {
        'at': datetime.now(timezone.utc).isoformat(), 'from': contract['gui'], 'to': gui}]
    updated['gui'] = gui
    return updated


def wait_launcher(process, timeout, trial, trial_dir):
    started = time.monotonic()
    while True:
        remaining = timeout-(time.monotonic()-started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(15., remaining))
        except subprocess.TimeoutExpired:
            run = latest_run(trial_dir)
            print(json.dumps({'type': 'evaluation_trial_wait', 'trial': trial,
                'wall_seconds': round(time.monotonic()-started, 1),
                'run': str(run) if run else None,
                'launcher_log': str(trial_dir/'launcher.log')}), flush=True)


def evaluate(output, contract, *, resume=False):
    records = []
    for trial in range(1, contract['trials']+1):
        trial_dir = output/'trials'/f'{trial:03d}'
        result = trial_dir/'result.json'
        scene = contract['scene_start']+trial-1
        if result.exists():
            record = json.loads(result.read_text())
            if record['trial'] != trial or record['scene_attempt'] != scene:
                raise ValueError('Saved trial identity changed')
            records.append(record)
            continue
        trial_dir.mkdir(parents=True, exist_ok=True)
        run = latest_run(trial_dir)
        recovered = None
        if resume and run is not None:
            try:
                recovered = score_run(run, contract['checkpoint_sha256'], scene)
            except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError):
                pass  # Incomplete/interrupted trials run again; old logs remain.
        if recovered is None:
            from storage_checks import ensure_free_space, GIB
            ensure_free_space(output, 3*GIB, 'running the next evaluation trial')
            command = [sys.executable, '-u', str(ROOT/'scripts/run_sfp0_multiview_gazebo.py'),
                '--checkpoint', str(output/'checkpoint.pt'),
                '--config', str(output/'inference_config.json'),
                '--results-dir', str(trial_dir/'runs'), '--device', contract['device'],
                '--amp', contract['amp'], '--gui', contract['gui'],
                '--renderer', contract['renderer'], '--scene-attempt', str(scene),
                '--real-time-factor', str(contract['real_time_factor']),
                '--max-task-seconds', str(contract['max_task_seconds']),
                '--max-wall-seconds', str(contract['max_wall_seconds'])]
            print(json.dumps({'type': 'evaluation_trial_start', 'trial': trial,
                'total': contract['trials'], 'scene_attempt': scene}), flush=True)
            # Append preserves diagnostics if an interrupted trial is retried.
            with (trial_dir/'launcher.log').open('a') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    status = wait_launcher(process, contract['max_wall_seconds']+120, trial, trial_dir)
                except KeyboardInterrupt:
                    stop_launcher(process)
                    save_reports(output, records, contract['trials'])
                    print('Interrupted trial remains pending; use --resume to continue.', flush=True)
                    return 130
                except subprocess.TimeoutExpired:
                    stop_launcher(process)
                    status = 124
            run = latest_run(trial_dir)
            try:
                if status != 0:
                    raise ValueError(f'Launcher exit code {status}; inspect {trial_dir/"launcher.log"}')
                if run is None:
                    raise ValueError('Launcher produced no run manifest')
                recovered = score_run(run, contract['checkpoint_sha256'], scene)
            except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
                recovered = {'outcome': 'error', 'success': False, 'run': str(run) if run else None,
                    'error': str(error), 'tier_1_score': None, 'tier_3_score': None,
                    'wall_seconds': None}
        record = {'trial': trial, 'scene_attempt': scene, **recovered}
        # Commit the official result before pruning optional heavy scoring bags.
        json_write(result, record)
        records.append(record)
        save_reports(output, records, contract['trials'])
        if record['outcome'] != 'error' and not contract['keep_scoring_bags']:
            from expert_batch import prune_scoring_bags
            prune_scoring_bags(Path(record['run']))
        if record['outcome'] == 'error':
            print('Stopped after infrastructure error; inspect launcher.log. '
                  'This attempt counts as an error; --resume continues with the next scene.', flush=True)
            return 2
    save_reports(output, records, contract['trials'])
    return 0


def main(policy_type='pace', default_model_root=None):
    if policy_type not in POLICY_FORMATS:
        raise ValueError('Unknown evaluation policy type')
    model_root = Path(default_model_root or f'/mnt/d/robot_models/aic_sfp0_{policy_type}_expert_v1')
    parser = argparse.ArgumentParser(
        description=f'Evaluate a frozen four-view {policy_type.upper()} checkpoint on reproducible Gazebo scenes.')
    parser.add_argument('--checkpoint', type=Path, default=model_root/'checkpoints/best.pt')
    parser.add_argument('--output-dir', type=Path, help='New directory on D:; specify it again with --resume')
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--scene-start', type=int, default=1, help='Trial i uses scene-start+i-1')
    parser.add_argument('--gui', choices=('true', 'false'), default=None,
                        help='New evaluation defaults to false; explicitly set to override display mode on resume')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--amp', choices=('bf16', 'fp16', 'none'), default='bf16')
    parser.add_argument('--renderer', choices=('hardware', 'software'), default='hardware')
    parser.add_argument('--real-time-factor', type=float, default=.5)
    parser.add_argument('--max-task-seconds', type=float, default=60)
    parser.add_argument('--max-wall-seconds', type=float, default=600)
    parser.add_argument('--keep-scoring-bags', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.trials < 1 or args.scene_start < 1 or args.scene_start+args.trials-1 >= 2**32
            or not math.isfinite(args.real_time_factor) or not 0 < args.real_time_factor <= 1
            or not math.isfinite(args.max_task_seconds) or args.max_task_seconds <= 0
            or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 10):
        parser.error('Use positive trials, positive scene IDs below 2**32, valid time limits and 0 < real-time-factor <= 1')
    if args.resume and args.output_dir is None:
        parser.error('--resume requires the previous --output-dir')
    if args.device == 'cpu':
        args.amp = 'none'
    output = (args.output_dir or model_root/'evaluations'/
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if not output.is_relative_to(Path('/mnt/d')):
        parser.error('Evaluation output/checkpoint must be on D:')
    if args.resume and not (output/'evaluation.json').is_file():
        parser.error('Missing evaluation.json; select the existing evaluation directory')
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error('Output is not empty; use --resume or choose another directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another evaluation process is using this directory')
        if args.resume:
            contract = json.loads((output/'evaluation.json').read_text())
            if contract.get('format') != evaluation_format(policy_type):
                parser.error(f'This is not a {policy_type.upper()} evaluation directory; keep ACT and PACE results separate')
            if digest_file(output/'checkpoint.pt') != contract['checkpoint_sha256']:
                parser.error('Frozen checkpoint or evaluation contract changed')
            if digest_file(output/'inference_config.json') != contract['inference_config_sha256']:
                parser.error('Frozen inference settings changed')
            updated = resume_gui(contract, args.gui)
            if updated != contract:
                print(f'GUI override: {contract["gui"]} -> {updated["gui"]}', flush=True)
                contract = updated
                json_write(output/'evaluation.json', contract)
        else:
            from run_sfp0_multiview_gazebo import snapshot_checkpoint
            from storage_checks import ensure_free_space, GIB
            ensure_free_space(output, 3*GIB)
            digest = snapshot_checkpoint(args.checkpoint, output/'checkpoint.pt')
            import torch
            payload = torch.load(output/'checkpoint.pt', map_location='cpu', weights_only=True)
            try:
                check_policy(payload, policy_type)
            except ValueError as error:
                parser.error(str(error))
            config_source = ROOT/'configs'/('sfp0_flow_multiview_gazebo.json' if policy_type in ('flow','residual_flow','stacked_residual_flow','alternating_residual_flow') else 'sfp0_multiview_gazebo.json')
            config = json.loads(config_source.read_text())
            json_write(output/'inference_config.json', config)
            args.gui = args.gui or 'false'
            contract = {'format': evaluation_format(policy_type), 'policy_type': policy_type,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source_checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest,
                'checkpoint_step': payload['step'], 'data_index_sha256': payload['data_index_sha256'],
                'inference_config_sha256': digest_file(output/'inference_config.json'),
                **{k: getattr(args, k) for k in ('trials', 'scene_start', 'gui', 'device', 'amp',
                    'renderer', 'real_time_factor', 'max_task_seconds', 'max_wall_seconds', 'keep_scoring_bags')}}
            del payload
            json_write(output/'evaluation.json', contract)
        print(f'Evaluation: {output}\nFrozen {policy_type.upper()} step {contract["checkpoint_step"]}; '
              f'{contract["trials"]} autonomous trials; GUI={contract["gui"]}; expert assistance disabled.', flush=True)
        try:
            return evaluate(output, contract, resume=args.resume)
        except KeyboardInterrupt:
            print('Stopped; completed trial results are preserved. Continue with --resume.', flush=True)
            return 130


if __name__ == '__main__':
    raise SystemExit(main())
