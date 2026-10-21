"""Timestamp alignment and bounded position commands for live ACT execution.

These helpers use neither ROS nor PyTorch so their timing contracts can be
checked without a simulator. Prediction row zero always means observation+dt.
"""
from collections import deque
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import numpy as np

from robot_bringup.demonstration_images import JOINT_NAMES


@dataclass
class Observation:
    stamp_ns: int
    images: dict
    qpos: np.ndarray


class ObservationBuffer:
    def __init__(self, cameras, slop_ns=20_000_000, state_age_ns=50_000_000):
        self.pending = {name: deque(maxlen=5) for name in cameras}
        self.states = deque(maxlen=200)
        self.last_image = {name: 0 for name in cameras}
        self.slop_ns, self.state_age_ns = slop_ns, state_age_ns
        self.latest = None

    def add_image(self, camera, stamp_ns, image):
        if stamp_ns > self.last_image[camera]:
            self.last_image[camera] = stamp_ns
            self.pending[camera].append((stamp_ns, image))
            self._align()

    def add_state(self, stamp_ns, names, positions):
        if stamp_ns <= 0 or (self.states and stamp_ns <= self.states[-1][0]):
            return
        if len(names) != len(positions) or len(set(names)) != len(names):
            return
        mapping = dict(zip(names, positions))
        if not all(name in mapping for name in JOINT_NAMES):
            return
        qpos = np.asarray([mapping[name] for name in JOINT_NAMES], dtype=np.float64)
        if np.isfinite(qpos).all():
            self.states.append((stamp_ns, qpos))
            self._align()

    def _align(self):
        while all(self.pending.values()):
            stamps = {name: queue[0][0] for name, queue in self.pending.items()}
            first = min(stamps, key=stamps.get)
            stamp = max(stamps.values())
            if stamp - stamps[first] > self.slop_ns:
                self.pending[first].popleft()
                continue
            if not self.states or self.states[-1][0] < stamp:
                return
            state = next((state for state in reversed(self.states) if state[0] <= stamp), None)
            images = {name: queue.popleft()[1] for name, queue in self.pending.items()}
            if state is not None and stamp - state[0] <= self.state_age_ns:
                self.latest = Observation(stamp, images, state[1].copy())


class TemporalEnsembler:
    def __init__(self, dt_ns, chunk_size, decay=.01, enabled=True):
        if dt_ns <= 0 or chunk_size < 1 or not np.isfinite(decay) or decay < 0:
            raise ValueError('Invalid action timing or ensemble decay')
        self.dt_ns, self.chunk_size = dt_ns, chunk_size
        self.decay, self.enabled = decay, enabled
        self.chunks = deque(maxlen=chunk_size + 1)

    def add(self, stamp_ns, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.chunk_size, 7) or not np.isfinite(actions).all():
            raise ValueError('Expected a finite K x 7 action chunk')
        if self.chunks and stamp_ns <= self.chunks[-1][0]:
            raise ValueError('Prediction observations must advance in time')
        self.chunks.append((stamp_ns, actions.copy()))

    def at(self, target_ns):
        # Only combine predictions of the SAME future time; never stale row 0.
        while self.chunks and self.chunks[0][0] + self.chunk_size * self.dt_ns < target_ns - 20_000_000:
            self.chunks.popleft()
        candidates, stamps = [], []
        for stamp, actions in self.chunks:
            offset = int(round((target_ns - stamp) / self.dt_ns))
            if (1 <= offset <= self.chunk_size and
                    abs(target_ns - (stamp + offset * self.dt_ns)) <= 20_000_000):
                candidates.append(actions[offset - 1])
                stamps.append(stamp)
        if not candidates:
            return None
        if not self.enabled:
            return candidates[-1].copy()
        # Original ACT weights older overlapping predictions more strongly.
        weights = np.exp(-self.decay * (np.asarray(stamps) - stamps[0]) / self.dt_ns)
        return np.average(candidates, axis=0, weights=weights)


@dataclass
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_urdf(cls, description):
        root = ET.fromstring(description)
        plugins = [p.text for p in root.findall('./ros2_control/hardware/plugin')]
        if 'gz_ros2_control/GazeboSimSystem' not in plugins:
            raise ValueError('ACT rollout requires the GazeboSimSystem robot description')
        lower, upper, velocity = [], [], []
        for name in JOINT_NAMES:
            joint = root.find(f"joint[@name='{name}']")
            if joint is None or joint.get('type') != 'revolute' or joint.find('mimic') is not None:
                raise ValueError('Missing independent revolute joint: ' + name)
            limits = joint.find('limit')
            lower.append(float(limits.get('lower')))
            upper.append(float(limits.get('upper')))
            velocity.append(float(limits.get('velocity')))
        result = cls(np.array(lower), np.array(upper), np.array(velocity))
        if (not np.isfinite([result.lower, result.upper, result.velocity]).all() or
                np.any(result.lower >= result.upper) or np.any(result.velocity <= 0)):
            raise ValueError('Invalid URDF joint limits')
        return result

    def bound(self, action, measured, seconds, arm_velocity=.6, gripper_velocity=.6):
        action, measured = np.asarray(action), np.asarray(measured)
        if (action.shape != (7,) or measured.shape != (7,) or
                not np.isfinite([action, measured]).all() or not np.isfinite(seconds) or seconds <= 0):
            raise ValueError('Invalid action, feedback or command duration')
        velocity = np.minimum(self.velocity, [arm_velocity] * 6 + [gripper_velocity])
        if not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise ValueError('Velocity bounds must be finite and positive')
        if np.any(measured < self.lower - .05) or np.any(measured > self.upper + .05):
            raise RuntimeError('Measured joint position exceeds URDF limits')
        desired = np.clip(action, self.lower, self.upper)
        desired = measured + np.clip(desired - measured, -velocity * seconds, velocity * seconds)
        return np.clip(desired, self.lower, self.upper)


def next_target_ns(now_ns, anchor_ns, dt_ns):
    """Next grid time, with >=10% of one period left for controller transport."""
    target = anchor_ns + ((now_ns - anchor_ns) // dt_ns + 1) * dt_ns
    return target + dt_ns if target - now_ns < dt_ns // 10 else target
