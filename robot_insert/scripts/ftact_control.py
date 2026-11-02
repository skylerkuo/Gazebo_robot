"""ROS-message conversion and bounded TCP commands, independent of ROS imports."""
from collections import deque

import cv2
import numpy as np

STATE_JOINT_NAMES = (
    "elbow_joint", "gripper/left_finger_joint", "shoulder_lift_joint",
    "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint",
)


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def ros_image_rgb(message):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(message.encoding)
    if channels is None:
        raise ValueError(f"Unsupported camera encoding: {message.encoding}")
    if message.height <= 0 or message.width <= 0 or message.step < message.width * channels:
        raise ValueError("Incomplete camera image")
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if len(data) != message.height * message.step:
        raise ValueError("Camera byte count does not match height and row stride")
    image = data.reshape(message.height, message.step)[:, :message.width * channels]
    image = image.reshape(message.height, message.width, channels)[..., :3]
    if message.encoding.startswith("bgr"):
        image = image[..., ::-1]
    # Match the source recorder's quarter-resolution RGB preprocessing.
    image = cv2.resize(image, (288, 256), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def observation_arrays(observation, joint_names=STATE_JOINT_NAMES, source_joints=None):
    # The image's Observation adapter renames the gripper joint. Use the
    # original /joint_states names to match the source dataset explicitly.
    joints = source_joints if source_joints is not None else observation.joint_states
    if len(joints.name) != len(joints.position) or len(set(joints.name)) != len(joints.name):
        raise ValueError("Invalid joint names/positions")
    positions = dict(zip(joints.name, joints.position))
    missing = set(joint_names) - positions.keys()
    if missing:
        raise ValueError(f"Missing source-state joints: {sorted(missing)}")
    state = np.asarray([positions[name] for name in joint_names], dtype=np.float32)
    wrench = observation.wrist_wrench.wrench
    force = np.asarray([wrench.force.x, wrench.force.y, wrench.force.z,
                        wrench.torque.x, wrench.torque.y, wrench.torque.z], dtype=np.float32)
    if not np.isfinite(state).all() or not np.isfinite(force).all() or not state.any():
        raise ValueError("Nonfinite or uninitialized state/wrench")
    result = {"observation.state": state, "observation.environment_state": force}
    for camera in ("left", "center", "right"):
        result[f"observation.images.{camera}_camera"] = ros_image_rgb(getattr(observation, camera + "_image"))
    if not result["observation.images.center_camera"].any():
        raise ValueError("Camera is still a black initialization frame")
    return result


def pose_array(pose):
    return np.asarray([pose.position.x, pose.position.y, pose.position.z,
                       pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)


def unit_pose(pose):
    value = np.asarray(pose, dtype=float).copy()
    if value.shape != (7,) or not np.isfinite(value).all() or np.linalg.norm(value[3:]) < 1e-8:
        raise ValueError("Invalid TCP pose")
    value[3:] /= np.linalg.norm(value[3:])
    return value


def bounded_pose(target, previous, minimum, maximum, translation_step, rotation_step):
    """Limit changes from the previous command; preserve all seven pose components."""
    target, previous = unit_pose(target), unit_pose(previous)
    if (target[:3] < minimum).any() or (target[:3] > maximum).any():
        raise ValueError(f"Predicted XYZ outside the configured dataset workspace: {target[:3].tolist()}")
    result = target.copy()
    delta = target[:3] - previous[:3]
    distance = np.linalg.norm(delta)
    if distance > translation_step:
        result[:3] = previous[:3] + delta * translation_step / distance
    dot = np.dot(previous[3:], target[3:])
    if dot < 0:
        target[3:] *= -1
        dot = -dot
    angle = 2 * np.arccos(np.clip(dot, 0, 1))
    if angle > rotation_step:
        fraction = rotation_step / angle
        half = angle / 2
        result[3:] = (np.sin((1-fraction)*half)*previous[3:] + np.sin(fraction*half)*target[3:]) / np.sin(half)
    else:
        result[3:] = target[3:]
    result[3:] /= np.linalg.norm(result[3:])
    return result, {"translation_limited": bool(distance > translation_step),
                    "rotation_limited": bool(angle > rotation_step)}


class TimedActionChunk:
    """Align next-tick pose labels with simulator time, skipping expired predictions.

    Element i predicts the command at reference_time + (i+1)/rate_hz.
    Replan halfway through the horizon; never restart at element zero on every tick.
    """
    def __init__(self, rate_hz=20, n_action_steps=10):
        if rate_hz <= 0 or n_action_steps < 1:
            raise ValueError('Invalid chunk execution settings')
        self.rate_hz, self.n_action_steps = rate_hz, n_action_steps
        self.chunk, self.reference_time = None, None

    def index(self, now):
        if self.reference_time is None:
            return None
        return max(0, int(np.floor((now - self.reference_time) * self.rate_hz + 1e-6)))

    def needs_refill(self, now):
        return self.chunk is None or self.index(now) >= min(len(self.chunk), self.n_action_steps)

    def replace(self, chunk, reference_time):
        value = np.asarray(chunk, dtype=float)
        if value.ndim != 2 or value.shape[1] != 7 or len(value) < self.n_action_steps:
            raise ValueError('Incomplete TCP action chunk')
        if not np.isfinite(value).all() or (np.linalg.norm(value[:, 3:], axis=1) < 1e-8).any():
            raise ValueError('Nonfinite or degenerate TCP chunk')
        self.chunk, self.reference_time = value.copy(), float(reference_time)

    def target(self, now):
        if self.chunk is None or self.index(now) >= len(self.chunk):
            raise ValueError('Action chunk expired; obtain a fresh prediction')
        index = self.index(now)
        return self.chunk[index].copy(), index


class ACTTemporalPoseEnsembler:
    """ACT Algorithm 2 with timestamp-aligned absolute TCP predictions.

    Predict every control step. Average only predictions for the same target
    tick with w_i=exp(-m*i), ordered OLDEST first, as in the original ACT code.
    Element i targets observation_time+(i+1)/rate_hz (the v2 training contract).
    Quaternion hemispheres are aligned before the component mean, then the
    result is normalized because the original joint-space actions are now poses.
    """
    def __init__(self, rate_hz=20, chunk_size=20, coefficient=.01):
        if not np.isfinite(rate_hz) or rate_hz <= 0 or chunk_size < 1:
            raise ValueError('Invalid ACT temporal ensemble dimensions')
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError('ACT temporal ensemble coefficient must be finite and nonnegative')
        self.rate_hz, self.chunk_size, self.coefficient = rate_hz, chunk_size, coefficient
        self.reset()

    def reset(self):
        self.origin = None
        self.history = deque(maxlen=self.chunk_size)
        self.last_observation_tick = None

    def add(self, chunk, observation_stamp, inference_id):
        value = np.asarray(chunk, dtype=np.float64).copy()
        if value.shape != (self.chunk_size, 7) or not np.isfinite(value).all():
            raise ValueError('Expected a finite complete ACT TCP action chunk')
        norms = np.linalg.norm(value[:, 3:], axis=1, keepdims=True)
        if (norms < 1e-8).any() or not np.isfinite(observation_stamp):
            raise ValueError('Invalid quaternion or observation timestamp')
        value[:, 3:] /= norms
        origin = float(observation_stamp) if self.origin is None else self.origin
        position = (observation_stamp - origin) * self.rate_hz
        tick = int(round(position))
        # Simulated camera stamps are on a 20 Hz grid. Reject misaligned input
        # instead of silently blending predictions for different physical times.
        if abs(position - tick) > .1:
            raise ValueError('Camera timestamp is not aligned with the ACT action time grid')
        if self.last_observation_tick is not None and tick <= self.last_observation_tick:
            raise ValueError('ACT requires a new observation tick for each prediction')
        self.origin = origin
        self.last_observation_tick = tick
        self.history.append((tick, float(observation_stamp), int(inference_id), value))

    def target(self, now):
        if self.origin is None or not np.isfinite(now) or now < self.origin:
            raise ValueError('No time-aligned ACT prediction is available')
        target_tick = int(np.floor((now - self.origin) * self.rate_hz + 1e-6)) + 1
        while self.history and self.history[0][0] + self.chunk_size < target_tick:
            self.history.popleft()
        rows = [(tick, stamp, prediction_id, target_tick - tick - 1,
                 chunk[target_tick - tick - 1])
                for tick, stamp, prediction_id, chunk in self.history
                if 0 <= target_tick - tick - 1 < self.chunk_size]
        if not rows:
            raise ValueError('All ACT predictions for this control time have expired')
        poses = np.stack([row[4] for row in rows])
        weights = np.exp(-self.coefficient * np.arange(len(rows), dtype=np.float64))
        weights /= weights.sum()
        aligned = poses.copy()
        opposite = (aligned[:, 3:] * aligned[0, 3:]).sum(axis=1) < 0
        aligned[opposite, 3:] *= -1
        result = unit_pose((aligned * weights[:, None]).sum(axis=0))
        details = {'ensemble_count': len(rows), 'ensemble_origin_time': self.origin,
            'ensemble_target_tick': target_tick,
            'ensemble_target_time': self.origin + target_tick / self.rate_hz,
            'ensemble_observation_stamps': [r[1] for r in rows],
            'ensemble_inference_ids': [r[2] for r in rows],
            'ensemble_source_indices': [r[3] for r in rows],
            'ensemble_source_poses': poses.tolist(), 'ensemble_weights': weights.tolist()}
        return result, details
