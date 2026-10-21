"""Read collector schema 3 without ROS; actions are strictly future measured qpos.

No target object pose, bin pose, task text, solution or final robot pose is used
as a policy input. Metadata is used only for filtering and execution boundaries.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np

from robot_bringup.demonstration_images import JOINT_NAMES

ACTION_SOURCE = 'future_measured_qpos'
DEFAULT_CAMERAS = ('scene', 'wrist', 'side')


@dataclass
class Episode:
    directory: Path
    stamps: np.ndarray
    qpos: np.ndarray
    image_paths: list
    # Exclusive end of the contiguous segment containing each observation.
    segment_ends: np.ndarray
    starts: np.ndarray

    def chunk(self, start, chunk_size):
        """q[t+1:t+1+K], padded only after a segment ends; never cross episodes."""
        stop = min(start + 1 + chunk_size, int(self.segment_ends[start]))
        future = self.qpos[start + 1:stop]
        if not len(future):
            raise ValueError('Observation has no continuous future state')
        actions = np.repeat(future[-1:], chunk_size, axis=0)
        actions[:len(future)] = future
        is_pad = np.arange(chunk_size) >= len(future)
        return actions, is_pad


def _relative_image(directory, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Image path must stay inside its episode')
    path = directory / path
    if not path.is_file():
        raise ValueError(f'Missing image: {path}')
    return path


def load_episode(directory, cameras=DEFAULT_CAMERAS, fps=10.0, timing_tolerance_ms=25.0):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'metadata.json').read_text())
    if (metadata.get('schema_version') != 3 or metadata.get('robot') != 'kinova_gen3_lite' or
            metadata.get('status') != 'placement_verified' or metadata.get('training_candidate') is not True or
            metadata.get('execution_success') is not True or
            metadata.get('image_check', {}).get('verified') is not True or
            metadata.get('placement_check', {}).get('verified') is not True):
        raise ValueError('Not a verified schema-3 Kinova RGB demonstration')
    status = metadata['task_status']
    start_ns, end_ns = int(status['execution_start_ns']), int(status['execution_end_ns'])
    if not 0 < start_ns < end_ns:
        raise ValueError('Invalid execution interval')
    if fps <= 0 or not 0 < timing_tolerance_ms < 500 / fps:
        raise ValueError('Invalid sampling frequency or timing tolerance')
    for name in cameras:
        camera = metadata.get('cameras', {}).get(name, {})
        if abs(float(camera.get('fps', 0)) - fps) > 1e-6:
            raise ValueError(f'{name}: recorded FPS differs from requested {fps}')
    stamps, states, paths = [], [], []
    previous = 0
    with (directory / 'frames.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            stamp = int(row['stamp_ns'])
            if stamp <= previous:
                raise ValueError('Frame timestamps must increase strictly')
            previous = stamp
            if not start_ns <= stamp <= end_ns:
                continue  # Exclude warm-up, planning and post-execution hold.
            names = row['joint_names']
            if len(names) != len(JOINT_NAMES) or set(names) != set(JOINT_NAMES):
                raise ValueError('Expected six arm joints and one gripper joint')
            state = row['joint_state']
            qpos = np.asarray(state['qpos'], dtype=np.float32)
            if qpos.shape != (7,) or not np.isfinite(qpos).all():
                raise ValueError('Invalid joint positions')
            age = stamp - int(state['stamp_ns'])
            image_stamps = [int(row['images'][name]['stamp_ns']) for name in cameras]
            if (not 0 <= age <= 50_000_000 or max(image_stamps) - min(image_stamps) > 20_000_000 or
                    any(not 0 <= stamp - value <= 20_000_000 for value in image_stamps)):
                raise ValueError('Noncausal, stale or unsynchronized observation')
            stamps.append(stamp)
            states.append(qpos[[names.index(name) for name in JOINT_NAMES]])
            paths.append([_relative_image(directory, row['images'][name]['path']) for name in cameras])
    if len(stamps) < 2:
        raise ValueError('Too few execution observations')
    stamps = np.asarray(stamps, dtype=np.int64)
    # Never teach that a 200/300 ms gap is a single 100 ms action. Keep each
    # continuous segment, masking chunks that reach a gap or the episode end.
    continuous = np.abs(np.diff(stamps) - round(1e9 / fps)) <= round(timing_tolerance_ms * 1e6)
    ends = np.empty(len(stamps), dtype=np.int64)
    ends[-1] = len(stamps)
    for index in range(len(stamps) - 2, -1, -1):
        ends[index] = ends[index + 1] if continuous[index] else index + 1
    starts = np.flatnonzero(continuous)
    if not len(starts):
        raise ValueError('No observations with a continuous next action')
    return Episode(directory, stamps, np.asarray(states, dtype=np.float32), paths, ends, starts)


def discover_episodes(root, cameras=DEFAULT_CAMERAS, fps=10.0, timing_tolerance_ms=25.0):
    """Accept a dataset root, one timestamped session, or one episode directory."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    candidates = ([root / 'metadata.json'] if (root / 'metadata.json').is_file()
                  else sorted(root.glob('episode_*/metadata.json')) +
                  sorted(root.glob('*/episode_*/metadata.json')))
    episodes, skipped = [], []
    for path in candidates:
        try:
            episodes.append(load_episode(path.parent, cameras, fps, timing_tolerance_ms))
        except (ValueError, KeyError, TypeError, OSError) as error:
            skipped.append({'episode': str(path.parent), 'reason': str(error)})
    if not episodes:
        raise ValueError(f'No usable successful RGB episodes in {root}; rejected {len(skipped)} episodes')
    return episodes, skipped


def split_episodes(episodes, val_fraction=0.1, seed=42):
    if len(episodes) < 2:
        raise ValueError('Need at least two verified episodes for separate train/validation sets')
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
    return shuffled[count:], shuffled[:count]


def normalization_stats(episodes):
    """Only training episodes contribute; each observation/next state counted once."""
    qpos = np.concatenate([episode.qpos[episode.starts] for episode in episodes]).astype(np.float64)
    actions = np.concatenate([episode.qpos[episode.starts + 1] for episode in episodes]).astype(np.float64)
    return {'qpos_mean': qpos.mean(0).tolist(), 'qpos_std': np.maximum(qpos.std(0), 1e-2).tolist(),
            'action_mean': actions.mean(0).tolist(), 'action_std': np.maximum(actions.std(0), 1e-2).tolist()}
