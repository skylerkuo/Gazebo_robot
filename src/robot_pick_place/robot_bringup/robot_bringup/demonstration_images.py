"""JPEG storage and timestamp alignment for the three ACT demonstration views."""
from collections import Counter, deque
import json


CAMERAS = {
    'scene': '/camera_head',
    'wrist': '/camera_wrist',
    'side': '/camera_side',
}
JOINT_NAMES = [f'joint_{index}' for index in range(1, 7)] + ['right_finger_bottom_joint']


class MultiViewImages:
    def __init__(self, directory, slop_ns=20_000_000, state_age_ns=50_000_000):
        self.directory = directory
        self.slop_ns = slop_ns
        self.state_age_ns = state_age_ns
        self.pending = {name: deque(maxlen=100) for name in CAMERAS}
        self.stamps = {name: [] for name in CAMERAS}
        self.states = deque(maxlen=200)
        self.synchronized_stamps = []
        self.dropped = Counter()
        self.calibrations = set()
        self.maximum_skew_ns = 0
        self.index = (directory / 'frames.jsonl').open('w')
        for name in CAMERAS:
            (directory / 'images' / name).mkdir(parents=True)
        (directory / 'calibration').mkdir()

    def add_calibration(self, name, data):
        if name not in self.calibrations:
            (self.directory / 'calibration' / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')
            self.calibrations.add(name)

    def add_image(self, name, stamp_ns, frame_id, width, height, jpeg):
        if stamp_ns <= 0 or (self.stamps[name] and stamp_ns <= self.stamps[name][-1]):
            self.dropped[name + '_duplicate_or_invalid_stamp'] += 1
            return False
        path = f'images/{name}/{stamp_ns:019d}.jpg'
        (self.directory / path).write_bytes(jpeg)
        self.stamps[name].append(stamp_ns)
        if len(self.pending[name]) == self.pending[name].maxlen:
            self.dropped[name + '_queue_overflow'] += 1
        self.pending[name].append(dict(stamp_ns=stamp_ns, path=path, frame_id=frame_id,
                                       width=width, height=height))
        self._align()
        return True

    def add_joint_state(self, stamp_ns, names, positions, velocities):
        if stamp_ns <= 0 or (self.states and stamp_ns <= self.states[-1]['stamp_ns']):
            return
        indices = {name: index for index, name in enumerate(names)}
        if not all(name in indices for name in JOINT_NAMES) or len(positions) != len(names):
            return
        state = {'stamp_ns': stamp_ns, 'qpos': [float(positions[indices[name]]) for name in JOINT_NAMES]}
        if len(velocities) == len(names):
            state['qvel'] = [float(velocities[indices[name]]) for name in JOINT_NAMES]
        self.states.append(state)
        self._align()

    def _align(self):
        while all(self.pending.values()):
            fronts = {name: queue[0] for name, queue in self.pending.items()}
            timestamps = {name: frame['stamp_ns'] for name, frame in fronts.items()}
            first = min(timestamps, key=timestamps.get)
            stamp = max(timestamps.values())
            skew = stamp - timestamps[first]
            if skew > self.slop_ns:
                self.pending[first].popleft()
                self.dropped[first + '_unsynchronized'] += 1
                continue
            # Wait for the state stream to reach this exposure time. Select a
            # causal state, never a future joint state, within the allowed age.
            if not self.states or self.states[-1]['stamp_ns'] < stamp:
                return
            state = next((s for s in reversed(self.states) if s['stamp_ns'] <= stamp), None)
            for queue in self.pending.values():
                queue.popleft()
            if state is None or stamp - state['stamp_ns'] > self.state_age_ns:
                self.dropped['missing_fresh_joint_state'] += 1
                continue
            row = {'stamp_ns': stamp, 'images': fronts, 'joint_names': JOINT_NAMES,
                   'joint_state': state, 'camera_skew_ns': skew,
                   'joint_state_age_ns': stamp - state['stamp_ns']}
            self.index.write(json.dumps(row, allow_nan=False) + '\n')
            self.index.flush()
            self.synchronized_stamps.append(stamp)
            self.maximum_skew_ns = max(self.maximum_skew_ns, skew)

    def quality(self, start_ns, end_ns):
        def coverage(stamps):
            selected = [t for t in stamps if start_ns <= t <= end_ns]
            if len(selected) < 2:
                return {'complete': False, 'frames': len(selected)}
            gap = max([selected[0] - start_ns, end_ns - selected[-1]] +
                      [b - a for a, b in zip(selected, selected[1:])])
            return {'complete': gap <= 350_000_000, 'frames': len(selected), 'maximum_gap_ns': gap}
        views = {name: coverage(stamps) for name, stamps in self.stamps.items()}
        aligned = coverage(self.synchronized_stamps)
        verified = (end_ns > start_ns > 0 and all(v['complete'] for v in views.values()) and
                    aligned['complete'] and self.calibrations == set(CAMERAS))
        return {'verified': verified, 'views': views, 'synchronized': aligned,
                'calibration_views': sorted(self.calibrations), 'dropped': dict(self.dropped),
                'maximum_camera_skew_ns': self.maximum_skew_ns,
                'reason': 'three views and joint states cover execution' if verified else
                          'missing calibration or gaps in images / aligned joint states'}

    def write_preview(self, start_ns, end_ns):
        import cv2
        rows = [json.loads(line) for line in (self.directory / 'frames.jsonl').read_text().splitlines()]
        if not rows:
            return
        target = start_ns + (end_ns - start_ns) // 4 if end_ns > start_ns > 0 else rows[-1]['stamp_ns']
        row = min(rows, key=lambda item: abs(item['stamp_ns'] - target))
        images = []
        for name, frame in row['images'].items():
            image = cv2.imread(str(self.directory / frame['path']))
            if image is None:
                return
            cv2.putText(image, f"{name}  t={frame['stamp_ns'] / 1e9:.2f}s", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            images.append(image)
        if not cv2.imwrite(str(self.directory / 'preview.jpg'), cv2.hconcat(images)):
            raise RuntimeError('Could not write three-view preview')

    def close(self):
        self.index.close()
