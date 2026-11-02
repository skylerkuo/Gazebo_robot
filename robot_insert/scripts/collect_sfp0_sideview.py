#!/usr/bin/env python3
"""Collect verified four-view SFP0 demonstrations in the official Gazebo simulator.

All recordings/logs/Parquet files go to D:. The official scene is copied, never
patched in place. Resume appends episodes and never overwrites existing data.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/dataset_python'))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sideview_data import (ARM_JOINTS, CAMERAS, LEGACY_JOINTS, TASK, camera_geometry,
    digest_file, export_episode, json_write, make_engine, make_robot_description, make_world,
    successful_score, validate_settings)

SHARE = ROOT / 'runtime/rootfs/ws_aic/install/share'
TASK_TEXT = 'Insert the SFP cable into nic_card_mount_0/sfp_port_0'


def refresh_metadata(output, manifest, settings):
    accepted = sorted([a for a in manifest['attempts'] if a.get('accepted')],key=lambda a:a['episode_index'])
    features = {f'observation.images.{name}_camera': {'dtype': 'image',
                'shape': [settings['record_height'], settings['record_width'], 3],
                'names': ['height', 'width', 'channel']} for name in CAMERAS}
    for name, names in [('observation.state', LEGACY_JOINTS), ('observation.arm_joint_positions', ARM_JOINTS),
                        ('observation.environment_state', ('Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz')),
                        ('observation.tcp_pose', ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')),
                        ('action', ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'))]:
        features[name] = {'dtype': 'float32', 'shape': [len(names)], 'names': list(names)}
    for name in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'simulation_time',
                 'action_time', 'camera_times', 'joint_time', 'wrench_time'):
        dtype = 'int64' if name in ('frame_index', 'episode_index', 'index', 'task_index') else (
            'float32' if name == 'timestamp' else 'float64')
        features[name] = {'dtype': dtype, 'shape': [4 if name == 'camera_times' else 1],
                          'names': list(CAMERAS) if name == 'camera_times' else None}
    episodes, offset = [], 0
    for a in accepted:
        n, ep = a['validation']['frames'], a['episode_index']
        episodes.append({'episode_index': ep, 'tasks': [TASK_TEXT], 'length': n,
            'data/chunk_index': 0, 'data/file_index': ep,
            'dataset_from_index': offset, 'dataset_to_index': offset+n,
            'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0})
        episodes[-1].update({f'stats/{feature}/{key}': value
            for feature, stats in a['validation'].get('numeric_stats', {}).items() for key,value in stats.items()})
        offset += n
    info = {'codebase_version': 'v3.0', 'robot_type': 'aic_controller', 'total_episodes': len(accepted),
        'total_frames': offset, 'total_tasks': 1, 'chunks_size': 1000, 'data_files_size_in_mb': 100,
        'video_files_size_in_mb': 200, 'fps': settings['fps'],
        'splits': {'train': f'0:{len(accepted)}'},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': None, 'features': features}
    # Train/VAL/test grouping is deferred until collection finishes; no random frame split.
    json_write(output/'meta/info.json', info)
    meta = output/'meta/episodes/chunk-000/file-000.parquet'
    meta.parent.mkdir(parents=True, exist_ok=True)
    if episodes:
        pq.write_table(pa.Table.from_pylist(episodes), meta.with_suffix('.parquet.partial'))
        meta.with_suffix('.parquet.partial').replace(meta)
    tasks = output/'meta/tasks.parquet'
    pq.write_table(pa.Table.from_pylist([{'task_index': 0, 'task': TASK_TEXT}]), tasks.with_suffix('.parquet.partial'))
    tasks.with_suffix('.parquet.partial').replace(tasks)
    json_write(output/'_collection/manifest.json', manifest)


def runtime_settings(settings, real_time_factor=None):
    """Wall-clock pacing is run-specific; simulation-time dataset semantics stay fixed."""
    result = dict(settings)
    if real_time_factor is not None:
        result['real_time_factor'] = real_time_factor
    validate_settings(result)
    return result


def run_batch(output, attempts, settings, gui, timeout, real_time_factor=None, *, official_expert=False):
    settings = runtime_settings(settings, real_time_factor)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    mode = 'DirectExpert' if official_expert else 'Sideview'
    name = f'{stamp}_{mode}_{attempts[0]:06d}_{os.getpid()}'
    run = output/'runs'/name
    run.mkdir(parents=True)
    engine_source = SHARE/'aic_engine/config/sample_config.yaml'
    engine, scenes = make_engine(yaml.safe_load(engine_source.read_text()), attempts, settings)
    (run/'engine_config.yaml').write_text(yaml.safe_dump(engine, sort_keys=False))
    make_world(SHARE/'aic_description/world/aic.sdf', run/'sideview_world.sdf', settings)
    make_robot_description(SHARE/'aic_description/urdf/ur_gz.urdf.xacro',
        SHARE/'aic_assets/models/Basler Camera/basler_camera_macro.xacro',
        run/'recording_robot.xacro', run/'recording_camera.xacro', settings,
        f'/workspace/results/{name}')
    bridge = yaml.safe_load((SHARE/'aic_bringup/config/ros_gz_bridge_config.yaml').read_text())
    for topic, ros_type, gz_type in [('image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
                                      ('camera_info', 'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo')]:
        bridge.append({'ros_topic_name': f'/side_camera/{topic}', 'gz_topic_name': f'/side_camera/{topic}',
            'ros_type_name': ros_type, 'gz_type_name': gz_type, 'direction': 'GZ_TO_ROS', 'lazy': True})
    (run/'bridge.yaml').write_text(yaml.safe_dump(bridge, sort_keys=False))
    runtime = dict(settings, raw_dir=f'/workspace/results/{name}/raw', trial_ids=list(engine['trials']),
                   camera_geometry=camera_geometry(settings['camera']))
    if official_expert:
        runtime['recording_mode'] = 'official_cheatcode_only'
    json_write(run/'settings.json', runtime)
    provenance = {'teacher': 'official aic_example_policies.ros.CheatCode', 'ground_truth_teacher_only': True,
        'trial_scenes': scenes, 'source_engine_sha256': digest_file(engine_source),
        'world_sha256': digest_file(run/'sideview_world.sdf'),
        'recorder_sha256': digest_file(ROOT/'policies/CollectSFP.py'),
        'recorder_helpers_sha256': digest_file(ROOT/'scripts/sideview_recorder.py'),
        'writer_sha256': digest_file(ROOT/'scripts/sideview_writer.py'),
        'data_helpers_sha256': digest_file(ROOT/'scripts/sideview_data.py')}
    json_write(run/'provenance.json', provenance)
    environment = dict(os.environ, INSERT_RESULTS_DIR=str(output/'runs'))
    command = [str(ROOT/'scripts/enter.sh'), '/bin/bash', '/workspace/scripts/session_sideview.sh',
               name, gui, str(max(1., timeout-10.))]
    print(json.dumps({'type': 'batch_start', 'run': str(run), 'attempts': attempts,
                      'real_time_factor': settings['real_time_factor']}), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    interrupted = False
    try:
        status = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        interrupted = True
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
        try:
            status = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            status = process.wait()
    json_write(run/'launcher.json', {'exit_code': status, 'interrupted_or_timed_out': interrupted,
        'wall_seconds': time.monotonic()-started, 'target_real_time_factor': settings['real_time_factor']})
    return run, status, interrupted


def import_batch(output, manifest, run, attempts, settings, status):
    scoring_path = run/'scoring/scoring.yaml'
    scores = yaml.safe_load(scoring_path.read_text()) if scoring_path.exists() else {}
    recorded_settings = json.loads((run/'settings.json').read_text())
    if any(recorded_settings.get(key) != value for key,value in settings.items()
           if key != 'real_time_factor'):
        raise ValueError('Recorded camera, timing or task settings differ from this collection')
    for attempt in attempts:
        trial = f'episode_{attempt:06d}'
        entry = {'attempt': attempt, 'trial': trial, 'run': str(run.relative_to(output)), 'accepted': False,
                 'runtime': {'real_time_factor': recorded_settings['real_time_factor']}}
        score = scores.get(trial, {})
        entry['official_score'] = score
        try:
            if status != 0 or not successful_score(score):
                raise ValueError('Official evaluator did not confirm successful insertion')
            summary = json.loads((run/'raw'/f'{trial}.json').read_text())
            if not summary['teacher_return'] or summary['writer_error']:
                raise ValueError('Teacher/recorder did not finish successfully')
            if not (run/'raw'/f'{trial}_calibration.json').is_file():
                raise ValueError('Missing camera calibration metadata')
            accepted = [a for a in manifest['attempts'] if a.get('accepted')]
            ep = len(accepted)
            offset = sum(a['validation']['frames'] for a in accepted)
            parquet = output/f'data/chunk-000/file-{ep:03d}.parquet'
            if parquet.exists():
                # Recover an export that finished just before its manifest commit.
                recovery = parquet.with_name(f'{parquet.stem}.{os.getpid()}.recovery.parquet')
                validation = export_episode(run/'raw'/f'{trial}.sqlite3', recovery, ep, offset, settings)
                same = digest_file(parquet)==validation['parquet_sha256']
                recovery.unlink()
                if not same:
                    raise ValueError(f'Existing {parquet} differs from this recording; refusing to overwrite')
            else:
                validation = export_episode(run/'raw'/f'{trial}.sqlite3', parquet, ep, offset, settings)
            entry.update(accepted=True, episode_index=ep, parquet=str(parquet.relative_to(output)),
                         validation=validation, recorder=summary)
        except Exception as error:
            entry['rejection_reason'] = f'{type(error).__name__}: {error}'
        manifest['attempts'].append(entry)
        refresh_metadata(output, manifest, settings)
        print(json.dumps({'type': 'episode_import', **entry}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/mnt/d/robot_datasets/aic_sfp0_multiview_v1'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sfp0_sideview.json')
    parser.add_argument('--episodes', type=int, default=20, help='Total number of accepted episodes, including earlier runs')
    parser.add_argument('--batch-size', type=int, default=5, help='Trials per simulator session')
    parser.add_argument('--real-time-factor', type=float,
        help='Override wall-clock simulation pacing for new sessions (0 < factor <= 1); saved per run')
    parser.add_argument('--max-attempts', type=int, default=40, help='Total attempt limit including resumed attempts')
    parser.add_argument('--max-wall-seconds', type=float, default=1800, help='Timeout per simulator session')
    parser.add_argument('--gui', choices=('true', 'false'), default='false')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--import-run', type=Path, help='Recover a finished session after an interrupted import')
    args = parser.parse_args()
    if args.episodes < 1 or args.batch_size < 1 or args.max_attempts < args.episodes or args.max_wall_seconds <= 0:
        parser.error('Use positive episode/batch/timeout values and max-attempts >= episodes')
    settings = json.loads(args.config.read_text())
    validate_settings(settings)
    try:
        runtime_settings(settings,args.real_time_factor)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir.expanduser().resolve()
    if not output.is_relative_to(Path('/mnt/d')):
        parser.error('Collection output must be on D: (/mnt/d/...)')
    if not (ROOT/'runtime/rootfs.ready').exists():
        parser.error('Prepare the official image first')
    output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'runtime/session.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another robot_insert Gazebo session is running')
    manifest_path = output/'_collection/manifest.json'
    if manifest_path.exists():
        if not args.resume:
            parser.error('Collection exists; use --resume to append without overwriting')
        manifest = json.loads(manifest_path.read_text())
        if manifest['settings'] != settings:
            parser.error('Collection settings differ; use another output directory')
    else:
        if any(output.iterdir()):
            parser.error('Output is nonempty without a collection manifest')
        manifest = {'format': settings['format'], 'settings': settings, 'task': TASK, 'attempts': [],
            'teacher': 'official CheatCode', 'model_observations_use_port_ground_truth': False,
            'action_frame': 'base_link', 'action_format': 'absolute TCP xyz+xyzw',
            'alignment': 'Each row stores the last accepted command at/before its observation timestamp. '
                         'Training should predict rows t+1 through t+20, without crossing gaps/episodes.',
            'camera_order': list(CAMERAS), 'split_status': 'Unsplit collection; group by episode before training.'}
    (output/'runs').mkdir(exist_ok=True)
    refresh_metadata(output, manifest, settings)
    if args.import_run:
        run = args.import_run.resolve()
        runtime = json.loads((run/'settings.json').read_text())
        done = {a['attempt'] for a in manifest['attempts']}
        attempts = [int(name.split('_')[1]) for name in runtime['trial_ids'] if int(name.split('_')[1]) not in done]
        status = json.loads((run/'launcher.json').read_text())['exit_code']
        import_batch(output, manifest, run, attempts, settings, status)
    while sum(a.get('accepted', False) for a in manifest['attempts']) < args.episodes:
        next_attempt = max((a['attempt'] for a in manifest['attempts']), default=-1)+1
        # A session may have completed before its import was interrupted. Never rerun over it.
        for existing in (output/'runs').glob('*/settings.json'):
            names = json.loads(existing.read_text())['trial_ids']
            if any(int(name.split('_')[1]) >= next_attempt for name in names):
                parser.error(f'Unimported session exists: {existing.parent}; recover with --import-run')
        remaining = args.episodes-sum(a.get('accepted', False) for a in manifest['attempts'])
        count = min(args.batch_size, remaining, args.max_attempts-next_attempt)
        if count <= 0:
            print('Attempt limit reached; inspect rejected episodes.', file=sys.stderr)
            return 1
        attempts = list(range(next_attempt, next_attempt+count))
        run, status, interrupted = run_batch(output, attempts, settings, args.gui, args.max_wall_seconds,
                                             args.real_time_factor)
        import_batch(output, manifest, run, attempts, settings, status)
        accepted = [a for a in manifest['attempts'] if a.get('accepted')]
        print(json.dumps({'type': 'collection_progress', 'accepted_episodes': len(accepted),
            'requested_episodes': args.episodes, 'frames': sum(a['validation']['frames'] for a in accepted)}), flush=True)
        if interrupted or status != 0 or not any(a.get('accepted') for a in manifest['attempts'][-count:]):
            print('Session failed or produced no usable episodes; inspect logs before retrying.', file=sys.stderr)
            return 1
    print(json.dumps({'type': 'collection_complete', 'output': str(output), 'episodes': args.episodes}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
