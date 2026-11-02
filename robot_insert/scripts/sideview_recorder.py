"""Synchronized four-view recording shared by teacher and human-assisted policies."""
from collections import Counter,deque
import json
from pathlib import Path
import threading
import time

import numpy as np
from aic_model_interfaces.msg import Observation
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo,Image,JointState
from std_msgs.msg import String
from tf2_ros import TransformException
from sideview_data import ARM_JOINTS,CAMERAS,LEGACY_JOINTS,TASK,json_write,latest_causal
from sideview_writer import EpisodeWriter


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def pose_array(value):
    return [value.position.x, value.position.y, value.position.z,
            value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w]


def transform_pose(value):
    return [value.translation.x, value.translation.y, value.translation.z,
            value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w]


def camera_info(message):
    return {'width': message.width, 'height': message.height, 'frame_id': message.header.frame_id,
            'k': list(message.k), 'p': list(message.p), 'r': list(message.r),
            'd': list(message.d), 'distortion_model': message.distortion_model}


class Recorder:
    def __init__(self, node, settings):
        self.node, self.settings = node, settings
        self.output = Path(settings['raw_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sides, self.joints, self.pending = deque(maxlen=6), deque(maxlen=100), deque()
        self.side_info = None
        self.active = None
        self.last_stamp = -1
        self.start_stamp = 0
        self.drop = Counter()
        self.observation_count = 0
        self.last_status_wall = 0
        self.last_error = None
        self.insertion_events = []
        self.recording_error = None
        group = ReentrantCallbackGroup()
        self.subscriptions = [
            node.create_subscription(Image, '/side_camera/image', self.side, qos_profile_sensor_data, callback_group=group),
            node.create_subscription(CameraInfo, '/side_camera/camera_info', self.info, qos_profile_sensor_data, callback_group=group),
            node.create_subscription(JointState, '/joint_states', self.joint, qos_profile_sensor_data, callback_group=group),
            node.create_subscription(Observation, '/observations', self.observation, 10, callback_group=group),
            node.create_subscription(String, '/scoring/insertion_event', self.insertion, 10, callback_group=group),
        ]
        self.timer = node.create_timer(.01, self.drain, callback_group=group)

    def side(self, message):
        with self.lock:
            self.sides.append((seconds(message.header.stamp), message))
        self.drain()

    def info(self, message):
        with self.lock:
            self.side_info = camera_info(message)

    def insertion(self, message):
        with self.lock:
            if self.active is not None:
                self.insertion_events.append({'receipt_simulation_time': self.node.get_clock().now().nanoseconds*1e-9,
                    'data': message.data, 'time_basis': 'callback receipt; event String has no source timestamp'})

    def joint(self, message):
        with self.lock:
            self.joints.append((seconds(message.header.stamp), message))

    def observation(self, message):
        with self.lock:
            stamp = seconds(message.center_image.header.stamp)
            if self.active is None or stamp < self.start_stamp or stamp <= self.last_stamp + 1e-8:
                return
            self.observation_count += 1
            if not self.pending or stamp > self.pending[-1][0] + 1e-8:
                self.pending.append((stamp, message, time.monotonic()))
        self.drain()

    def begin(self, task):
        name = task.id
        if name not in self.settings['trial_ids']:
            raise ValueError(f'Unexpected task ID: {name}')
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Another collection episode is active')
            self.start_stamp = self.node.get_clock().now().nanoseconds * 1e-9
            self.last_stamp = -1
            self.pending.clear()
            self.sides.clear();self.joints.clear()
            self.drop.clear()
            self.last_error=None;self.last_status_wall=0
            self.observation_count = 0
            self.insertion_events = []
            self.recording_error = None
            self.active = EpisodeWriter(self.output, name, self.settings)

    def command(self, message, *, source='teacher', segment=0, human_active=False):
        if message.header.frame_id != 'base_link':
            raise ValueError('Teacher command must use base_link')
        pose = np.asarray(pose_array(message.pose), dtype=float)
        norm = np.linalg.norm(pose[3:])
        if not np.isfinite(pose).all() or norm < 1e-6:
            raise ValueError('Nonfinite teacher command')
        pose[3:] /= norm
        with self.lock:
            if self.active is not None:
                self.active.put(('command', {'simulation_time': self.node.get_clock().now().nanoseconds*1e-9,
                    'header_time': seconds(message.header.stamp), 'pose': pose.tolist(),
                    'control_source':source,'control_segment':int(segment),'human_active':bool(human_active)}))

    def drain(self):
        with self.lock:
            if self.active is None:
                return
            while self.pending:
                stamp, observation, arrived = self.pending[0]
                reason = 'missing_side_image'
                side = latest_causal(self.sides, stamp, self.settings['camera_tolerance_s'])
                joint = latest_causal(self.joints, stamp, self.settings['state_tolerance_s'])
                try:
                    if side is None or self.side_info is None:
                        raise ValueError(reason)
                    reason = 'missing_joint_state'
                    if joint is None:
                        raise ValueError(reason)
                    reason = 'missing_tcp_tf'
                    tcp = self.node._tf_buffer.lookup_transform('base_link', 'gripper/tcp',
                                                                 Time.from_msg(observation.center_image.header.stamp))
                    reason = 'invalid_frame'
                    images = [observation.left_image, observation.center_image, observation.right_image, side[1]]
                    camera_times = {name: seconds(msg.header.stamp) for name, msg in zip(CAMERAS, images)}
                    if any(not -1e-8 <= stamp-value <= self.settings['camera_tolerance_s']+1e-8
                           for value in camera_times.values()):
                        raise ValueError('camera_skew')
                    positions = dict(zip(joint[1].name, joint[1].position))
                    if len(positions) != len(joint[1].name):
                        raise ValueError('duplicate_joint_names')
                    wrench = observation.wrist_wrench
                    wrench_time = seconds(wrench.header.stamp)
                    if not -1e-8 <= stamp-wrench_time <= self.settings['state_tolerance_s']+1e-8:
                        raise ValueError('stale_force')
                    w = wrench.wrench
                    record = {'simulation_time': stamp, 'camera_times': camera_times,
                        'joint_time': joint[0], 'wrench_time': wrench_time,
                        'state': [positions[name] for name in LEGACY_JOINTS],
                        'arm_joint_positions': [positions[name] for name in ARM_JOINTS],
                        'gripper_position': positions['gripper/left_finger_joint'],
                        'wrench': [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z],
                        'tcp_pose': transform_pose(tcp.transform)}
                    if self.last_stamp < 0:
                        infos = {name: camera_info(getattr(observation, name+'_camera_info'))
                                 for name in CAMERAS[:3]}
                        infos['side'] = self.side_info
                        extrinsics = {}
                        for name in CAMERAS[:3]:
                            tf = self.node._tf_buffer.lookup_transform('gripper/tcp', name+'_camera/optical', Time())
                            extrinsics[name] = transform_pose(tf.transform)
                        base_from_world = self.node._tf_buffer.lookup_transform('base_link', 'world', Time())
                        json_write(self.output / f'{self.active.name}_calibration.json',
                                   {'source_camera_info': infos, 'record_width': self.settings['record_width'],
                                    'record_height': self.settings['record_height'],
                                    'tcp_from_wrist_optical_xyz_xyzw': extrinsics,
                                    'base_from_world_xyz_xyzw': transform_pose(base_from_world.transform),
                                    'side_camera': self.settings['camera'],
                                    'side_geometry': self.settings['camera_geometry']})
                    self.active.put(('frame', record, images))
                    self.last_stamp = stamp
                    self.pending.popleft()
                except (ValueError, KeyError, TransformException) as error:
                    self.last_error = str(error)
                    # Wait for transport order / TF interpolation without choosing future sensor values.
                    if time.monotonic()-arrived < .5 and len(self.pending) < 5:
                        break
                    self.drop[str(error) if isinstance(error, ValueError) else reason] += 1
                    self.pending.popleft()
                except Exception as error:
                    self.drop[f'writer_error:{type(error).__name__}'] += 1
                    self.node.get_logger().error(f'Recording failure: {error}')
                    self.pending.popleft()
            if time.monotonic()-self.last_status_wall > 5:
                self.last_status_wall = time.monotonic()
                json_write(self.output/f'{self.active.name}_status.json',
                    {'observations':self.observation_count,'written_frames':self.active.frames,
                     'written_commands':self.active.commands,'queued':self.active.queue.qsize(),
                     'pending':len(self.pending),'drops':dict(self.drop),'last_error':self.last_error,
                     'latest_frame_time':self.last_stamp,
                     'latest_side_time':self.sides[-1][0] if self.sides else None})

    def finish(self, teacher_return, *, controller='teacher'):
        self.drain()
        with self.lock:
            writer, self.active = self.active, None
            pending = len(self.pending)
            self.pending.clear()
        writer.finish()
        summary = {'task': TASK, 'teacher_return': bool(teacher_return), 'frames': writer.frames,
                   'raw_format': 'buffered-png-pack-v1',
                   'commands': writer.commands, 'dropped_frames': dict(self.drop),
                   'pending_at_end': pending, 'writer_error': writer.failure or self.recording_error,
                   'diagnostic_insertion_events': self.insertion_events,
                   'success_requires_official_score': True}
        if controller!='teacher':
            summary.update(controller=controller,policy_return=bool(teacher_return),teacher_return=None)
        json_write(self.output / f'{writer.name}.json', summary)
        self.node.get_logger().info(f'COLLECTION_EPISODE {writer.name} {json.dumps(summary)}')
        return summary

