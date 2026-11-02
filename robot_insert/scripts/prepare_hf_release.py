#!/usr/bin/env python3
"""Freeze a portable, dataset-only public release on D: (no upload)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'tools/dataset_python'))
SOURCE = Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1')
OUTPUT = Path('/mnt/d/robot_datasets/_hf_release/aic-cable-insertion-gazebo-sfp0')
REPO = 'Kuoskyler/aic-cable-insertion-gazebo-sfp0'
UPSTREAM_REVISION = '6546127c1c15d5786f59c91400f5fc5ad6894448'
ORIGINAL_REVISION = '9d2ac51be268ef27cf2c2d0e073c7d9bb9acbe2c'
IMAGE = 'ghcr.io/intrinsic-dev/aic/aic_eval@sha256:253b99a4aa6e0e4fc0cf8c1cfd0269e64aa86411c2bd585c95a1732d4a56a165'
PRIVATE_PREFIXES = ('/mnt/', '/home/', '/workspace/results/', 'C:\\', 'D:\\')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assert_portable(value):
    """Reject private host paths rather than publishing silently redacted metadata."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert_portable(k)
            assert_portable(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_portable(item)
    elif isinstance(value, str) and any(p in value for p in PRIVATE_PREFIXES):
        raise ValueError('Release metadata contains a local host path')


def select_entries(manifest):
    from sideview_data import successful_score
    result = sorted((e for e in manifest['attempts'] if e.get('accepted')),
                    key=lambda e: e['episode_index'])
    if not result or len({e['episode_index'] for e in result}) != len(result):
        raise ValueError('Empty or duplicate episode selection')
    for e in result:
        if (e.get('operator_discarded') or e.get('test_run')
                or not successful_score(e['official_score'])
                or e['recorder'].get('writer_error')
                or not e['recorder'].get('policy_return')):
            raise ValueError('An accepted episode lacks successful official scoring/recording')
        if e['recorder']['controller'] not in ('official_cheatcode_only', 'model_and_official_expert'):
            raise ValueError('Unknown controller provenance')
        relative = Path(e['parquet'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.parquet':
            raise ValueError('Unsafe episode path')
    return result


def append_membership(previous, current):
    """An update may add episodes, but cannot replace or remove published data."""
    if previous.get('repo_id') != REPO or current.get('repo_id') != REPO:
        raise ValueError('Update snapshots must belong to the requested repository')
    before={f['episode_index']:f for f in previous['files']}
    after={f['episode_index']:f for f in current['files']}
    if len(before)!=len(previous['files']) or len(after)!=len(current['files']):
        raise ValueError('Duplicate episode IDs in release snapshot')
    if any(len({f['path'] for f in plan['files']})!=len(plan['files']) for plan in (previous,current)):
        raise ValueError('Duplicate episode paths in release snapshot')
    if any(after.get(ep)!=item for ep,item in before.items()):
        raise ValueError('An update cannot remove or change an existing episode')
    return [f for f in current['files'] if f['episode_index'] not in before]


def freeze(source, output, previous_release=None):
    import pyarrow.parquet as pq
    import yaml
    source, output = source.resolve(), output.resolve()
    if not all(p.is_relative_to(Path('/mnt/d')) for p in (source, output)):
        raise ValueError('Keep release data and staging on D:')
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use an independent release directory')
    manifest = json.loads((source / '_collection/manifest.json').read_text())
    entries = select_entries(manifest)
    membership = [{'episode_index': e['episode_index'], 'path': e['parquet'],
                   'sha256': e['validation']['parquet_sha256'],
                   'bytes': (source / e['parquet']).stat().st_size} for e in entries]
    fingerprint = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    previous_plan=None
    if previous_release is not None:
        previous_release=Path(previous_release).resolve()
        if not previous_release.is_relative_to(Path('/mnt/d')) or previous_release==output:
            raise ValueError('Use a separate previous release directory on D:')
        previous_plan=json.loads((previous_release/'meta/snapshot.json').read_text())
        append_membership(previous_plan, {'repo_id':REPO,'files':membership})
    plan_path = output / 'meta/snapshot.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['membership_sha256'] != fingerprint:
            raise ValueError('Source membership changed; choose a fresh release directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Nonempty destination has no frozen snapshot')
        plan = {'repo_id': REPO, 'created_at': datetime.now(timezone.utc).isoformat(),
                'membership_sha256': fingerprint, 'files': membership,
                'hash_origin': 'Recorded SHA-256 of each immutable source Parquet; upload verification compares remote LFS hashes.'}
        dump(plan_path, plan)
    episodes, total_sources = [], Counter()
    for number, e in enumerate(entries, 1):
        original, target = source / e['parquet'], output / e['parquet']
        if original.is_symlink():
            raise ValueError('Expected immutable regular Parquet')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(original, target):
                raise ValueError('Existing staging file is not the frozen source hardlink')
        else:
            # Same NTFS volume: no second 47 GiB copy, and no change to source bytes.
            os.link(original, target)
        parquet = pq.ParquetFile(target)
        if parquet.metadata.num_rows != e['validation']['frames']:
            raise ValueError('Parquet row count differs from accepted recording')
        sources = Counter(parquet.read(columns=['action_source']).column(0).to_pylist())
        if dict(sources) != e['validation']['control_source_counts']:
            raise ValueError('Action provenance differs from accepted recording')
        total_sources.update(sources)
        ep = e['episode_index']
        run = Path(e['run'])
        task_id = e.get('recording_task_id', 'task_1')
        calibration_path = f'calibration/episode-{ep:06d}.json'
        calibration = json.loads((run / 'raw' / f'{task_id}_calibration.json').read_text())
        engine = yaml.safe_load((run / 'engine_config.yaml').read_text())
        scene = engine['trials'][e['trial']]['scene']
        scene_path = f'environment/scenes/episode-{ep:06d}.json'
        settings = json.loads((run / 'settings.json').read_text())
        control_keys = ('rate_hz', 'n_action_steps', 'temporal_ensemble_coeff', 'stiffness', 'damping',
                        'expert_after_seconds', 'expert_z_stiffness', 'expert_xy_stiffness',
                        'max_translation_step_m', 'max_rotation_step_rad', 'max_force_n', 'max_torque_nm')
        item_keys = ('episode_index', 'attempt', 'trial', 'recording_type', 'controller_version',
                     'official_score', 'recorder', 'validation', 'filtered_expert_anchor_frames',
                     'expert_anchor_frames', 'expert_validation_frames', 'rollout_train_anchor_frames',
                     'idle_filter', 'sampling_report', 'rollout_sampling_report', 'rollout_filter',
                     'model_motion_preserved', 'outcome', 'validation_eligible')
        item = {k: e[k] for k in item_keys if k in e}
        item.update(data_path=e['parquet'], calibration_path=calibration_path, scene_path=scene_path,
                    episode_type='direct' if e['recorder']['controller'] == 'official_cheatcode_only' else 'assisted',
                    control_settings={k: settings[k] for k in control_keys if k in settings})
        for value in (calibration, scene, item):
            assert_portable(value)
        dump(output / calibration_path, calibration)
        dump(output / scene_path, scene)
        episodes.append(item)
        if number % 25 == 0 or number == len(entries):
            print(json.dumps({'type': 'release_episode', 'prepared': number, 'total': len(entries)}), flush=True)
    (output / 'meta/episodes.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in episodes))
    info = json.loads((source / 'meta/info.json').read_text())
    info.update(format='aic-four-view-parquet-release-v1', codebase_version='custom',
                compatibility='Hugging Face Datasets/Parquet; LeRobot-style features, not a certified native LeRobot v3 export',
                total_episodes=len(entries), total_frames=sum(e['validation']['frames'] for e in entries))
    assert_portable(info)
    dump(output / 'meta/info.json', info)
    shutil.copyfile(source / 'meta/tasks.parquet', output / 'meta/tasks.parquet')
    types = Counter(e['episode_type'] for e in episodes)
    stats = {'episodes': len(entries), 'frames': info['total_frames'], 'fps': info['fps'],
             'episode_types': dict(types), 'assisted_with_complete_expert_windows': sum(
                 e['episode_type'] == 'assisted' and bool(e.get('expert_validation_frames')) for e in episodes),
             'action_source_counts': dict(total_sources), 'parquet_bytes': sum(e['bytes'] for e in membership),
             'task': manifest['task'], 'collection_settings': manifest['settings'],
             'upstream_aic_revision': UPSTREAM_REVISION, 'official_runtime_image': IMAGE,
             'original_community_dataset': 'ivogeorg/aic-cable-insertion-gazebo-v3',
             'original_community_dataset_revision': ORIGINAL_REVISION,
             'split_note': 'train is the unsplit storage pool; split by complete episode for evaluation.'}
    update_note=''
    if previous_plan is not None:
        added=append_membership(previous_plan,plan)
        added_ids={f['episode_index'] for f in added}
        added_entries=[e for e in entries if e['episode_index'] in added_ids]
        stats['update']={'previous_episodes':len(previous_plan['files']), 'added_episodes':len(added),
            'added_frames':sum(e['validation']['frames'] for e in added_entries),
            'previous_membership_sha256':previous_plan['membership_sha256']}
        kind='controller-only ' if added_entries and all(
            e['recorder']['controller']=='official_cheatcode_only' for e in added_entries) else ''
        update_note=(f"## Latest update\n\nAdded {len(added)} {kind}episodes "
            f"({stats['update']['added_frames']:,} frames). The previous "
            f"{len(previous_plan['files'])} episodes are unchanged.\n")
    assert_portable(stats)
    dump(output / 'meta/release.json', stats)
    environment(output, Path(entries[-1]['run']), previous_release=previous_release)
    templates = PROJECT / 'publishing'
    text = (templates / 'DATASET_CARD.md').read_text()
    for key, value in {'EPISODES': stats['episodes'], 'FRAMES': f"{stats['frames']:,}",
                       'DIRECT': types['direct'], 'ASSISTED': types['assisted'],
                       'RECOVERY': stats['assisted_with_complete_expert_windows'],
                       'GIB': f"{stats['parquet_bytes'] / 2**30:.2f}", 'UPDATE_NOTE':update_note}.items():
        text = text.replace('{{' + key + '}}', str(value))
    (output / 'README.md').write_text(text)
    shutil.copyfile(PROJECT / 'upstream/LICENSE', output / 'LICENSE')
    for name in ('NOTICE', 'CITATION.cff'):
        shutil.copyfile(templates / name, output / name)
    # All publishable files are explicitly enumerated; never upload auth/cache/checkpoints.
    inventory = []
    for p in sorted(output.rglob('*')):
        if not p.is_file() or '.cache' in p.parts:
            continue
        relative = p.relative_to(output).as_posix()
        if relative in ('meta/files.json', 'meta/READY.json'):
            continue
        source_hash = next((f['sha256'] for f in membership if f['path'] == relative), None)
        inventory.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': source_hash or digest(p)})
    dump(output / 'meta/files.json', inventory)
    dump(output / 'meta/READY.json', {'membership_sha256': fingerprint, 'files': len(inventory), 'complete': True})
    print(json.dumps({'type': 'release_ready', 'output': str(output), **stats}), flush=True)
    return stats


def environment(output, representative_run, previous_release=None):
    target = output / 'environment'
    archive = target / 'aic_assets_and_description.tar.gz'
    if not archive.exists() and previous_release is not None:
        os.link(previous_release/'environment'/archive.name, archive)
    if not archive.exists():
        with tarfile.open(archive.with_suffix('.partial'), 'w:gz', compresslevel=3) as tar:
            for name in ('aic_assets', 'aic_description', 'LICENSE'):
                tar.add(PROJECT / 'upstream' / name, arcname=name, recursive=True)
        archive.with_suffix('.partial').replace(archive)
    for name in ('recording_camera.xacro', 'recording_robot.xacro', 'sideview_world.sdf', 'bridge.yaml'):
        if previous_release is not None:
            shutil.copyfile(previous_release/'environment'/name, target/name)
            continue
        text = (representative_run / name).read_text()
        if name == 'recording_robot.xacro':
            text = text.replace(f'/workspace/results/{representative_run.name}/recording_camera.xacro',
                                'recording_camera.xacro')
        assert_portable(text)
        (target / name).write_text(text)
    shutil.copyfile(PROJECT / 'publishing/ENVIRONMENT.md', target / 'README.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=SOURCE)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--previous-release', type=Path,
        help='Append episodes to this frozen release, preserving existing data and environment assets')
    a = p.parse_args()
    freeze(a.dataset_root, a.output_dir, a.previous_release)


if __name__ == '__main__':
    main()
