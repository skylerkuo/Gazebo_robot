#!/usr/bin/env python3
"""Index the first SFP task without copying the source RGB parquet files."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from ftact import CAMERAS, STATE, WRENCH, INSERT_ROOT, json_write, read_data_contract
from prepare_dataset import matrix, numeric_stats
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

DEFAULT_INDEX = Path('/mnt/d/robot_datasets/aic_sfp0_v2/index.json')


def numeric_boundary(state, action, timestamps):
    """A simulator teleport is much larger than ordinary joint motion in this recording."""
    joint = np.flatnonzero(np.linalg.norm(np.diff(state, axis=0), axis=1) > .2) + 1
    home = np.abs(action[1:, 3]) > np.cos(np.deg2rad(3) / 2)
    command = np.flatnonzero((np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1) > .12) & home) + 1
    gaps = np.flatnonzero(np.abs(np.diff(timestamps) - .05) > 1e-4) + 1
    end = min([len(state), *joint.tolist(), *command.tolist(), *gaps.tolist()])
    return int(end), {'joint_reset': joint.tolist(), 'command_reset': command.tolist(),
                      'timestamp_gap': gaps.tolist()}


def visual_boundary(images, limit, threshold=25.):
    """Stop at the first abrupt whole-view change; ignore initial black camera frames."""
    previous = None
    start = 0
    for i in range(min(len(images), limit + 1)):
        value = np.asarray(Image.open(io.BytesIO(images[i].as_py()['bytes']))
                           .convert('RGB').resize((72, 64)), dtype=np.float32)
        if previous is None and value.max() == 0:
            start = i + 1
            continue
        if previous is not None:
            difference = float(np.abs(value - previous).mean())
            if difference > threshold:
                return start, i, difference
        previous = value
    return start, limit, None


def balanced_anchors(segments, count, seed):
    """Round-robin random unique anchors, so long recordings do not dominate VAL."""
    rng = np.random.default_rng(seed)
    queues = [rng.permutation(np.arange(s['start'], s['end'] - 20)).tolist() for s in segments]
    order = rng.permutation(len(segments)).tolist()
    result = []
    while len(result) < count:
        progressed = False
        for i in order:
            if queues[i]:
                result.append([segments[i]['episode_index'], int(queues[i].pop())])
                progressed = True
                if len(result) == count:
                    break
        if not progressed:
            raise ValueError(f'Only {len(result)} unique validation anchors; requested {count}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path,
        default=Path('/mnt/d/robot_datasets/aic_cable_insertion_gazebo_v3_act'))
    parser.add_argument('--output', type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Index already exists: {output}; use a different --output to rebuild')
    _, _, source_contract = read_data_contract(root)
    mapping = json.loads((root / '_preparation/episode_mapping.json').read_text())
    segments, excluded = [], []
    training = {key: [] for key in (STATE, WRENCH, 'action')}
    thumbnails = []
    seen_sources = set()
    for number, record in enumerate(sorted(mapping, key=lambda m: (m['source_episode_index'], m['source_frame_start']))):
        source = record['source_episode_index']
        if source in seen_sources:
            excluded.append({'source_episode_index': source, 'reason': 'later_valid_run'})
            continue
        seen_sources.add(source)
        episode = record['episode_index']
        path = root / f'data/chunk-000/file-{episode:03d}.parquet'
        table = pq.read_table(path, columns=[STATE, WRENCH, 'action', 'timestamp', CAMERAS[1]])
        values = {key: matrix(table[key], width) for key, width in ((STATE, 6), (WRENCH, 6), ('action', 7))}
        candidate, reasons = numeric_boundary(values[STATE], values['action'],
                                               matrix(table['timestamp'], 1).ravel())
        start, visual_end, change = visual_boundary(table[CAMERAS[1]], candidate)
        boundary = min(candidate, visual_end)
        # RGB, joint state and the command recorder need not update on the same tick.
        end = max(start, boundary - 2) if boundary < len(table) else boundary
        if end - start <= 20:
            excluded.append({'source_episode_index': source, 'reason': 'too_short_after_task_boundary'})
            continue
        segment = {'episode_index': episode, 'source_episode_index': source,
            'split': record['split'], 'source_frame_start': record['source_frame_start'],
            'episode_length': len(table), 'start': start, 'end': end,
            'anchor_count': end - start - 20, 'numeric_boundary': candidate,
            'visual_boundary': visual_end, 'visual_change_mae_0_255': change, **reasons}
        segments.append(segment)
        if record['split'] == 'train':
            for key in training:
                training[key].append(values[key][start:end])
        # Every selected source gets start / terminal / first excluded images for review.
        strip = Image.new('RGB', (144 * 3, 144), 'white')
        draw = ImageDraw.Draw(strip)
        for x, frame in enumerate((start, end - 1, min(boundary, len(table) - 1))):
            im = Image.open(io.BytesIO(table[CAMERAS[1]][frame].as_py()['bytes'])).resize((144, 128))
            strip.paste(im, (x * 144, 0))
            draw.text((x * 144, 128), f'{source}:{frame + record["source_frame_start"]}', fill='black')
        thumbnails.append(strip)
        if (number + 1) % 20 == 0:
            print(json.dumps({'scanned': number + 1, 'kept': len(segments)}), flush=True)
    stats = {key: numeric_stats(np.concatenate(values)) for key, values in training.items()}
    # ImageNet normalization matches the pretrained visual backbone.
    for key in CAMERAS:
        stats[key] = {'mean': np.array([.485, .456, .406]).reshape(3, 1, 1).tolist(),
                      'std': np.array([.229, .224, .225]).reshape(3, 1, 1).tolist()}
    # Prevent nearly constant quaternion/joint channels from magnifying recorder noise.
    for key, floor in ((STATE, [.001] * 6), (WRENCH, [.5] * 3 + [.05] * 3),
                       ('action', [.01] * 7)):
        stats[key]['raw_std'] = stats[key]['std']
        stats[key]['std'] = np.maximum(stats[key]['std'], floor).tolist()
        stats[key]['std_floor'] = floor
    validation = balanced_anchors([s for s in segments if s['split'] == 'validation'], 1200, 10042)
    counts = {split: {'segments': sum(s['split'] == split for s in segments),
        'frames': sum(s['end'] - s['start'] for s in segments if s['split'] == split),
        'anchors': sum(s['anchor_count'] for s in segments if s['split'] == split)}
        for split in ('train', 'validation', 'test')}
    manifest = {'format': 'aic-sfp0-index-v2', 'dataset_root': str(root),
        'task': {'trial': 'trial_1', 'plug': 'sfp_tip', 'target_module': 'nic_card_mount_0', 'port': 'sfp_port_0'},
        'source_contract': source_contract, 'fps': 20, 'chunk_size': 20, 'action_offset': 1,
        'boundary_method': 'first numeric reset or RGB MAE>25; 2-frame pre-reset guard',
        'label_limitation': 'Task identity inferred from first-task order and resets; source provides no per-frame scene or success labels.',
        'segments': segments, 'excluded': excluded, 'counts': counts,
        'validation_anchors': validation, 'stats': stats}
    manifest['index_content_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    json_write(output, manifest)
    gallery = output.parent / 'boundary_review'
    gallery.mkdir(exist_ok=True)
    for page, offset in enumerate(range(0, len(thumbnails), 30)):
        sheet = Image.new('RGB', (432 * 3, 144 * 10), 'white')
        for local, thumb in enumerate(thumbnails[offset:offset + 30]):
            sheet.paste(thumb, ((local % 3) * 432, (local // 3) * 144))
        sheet.save(gallery / f'page_{page:02d}.jpg', quality=85)
    print(json.dumps({'index': str(output), 'counts': counts, 'validation_samples': len(validation),
                      'copied_rgb_files': 0, 'index_content_sha256': manifest['index_content_sha256']}, indent=2))


if __name__ == '__main__':
    main()
