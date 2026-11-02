"""Exercise the real Gazebo policy loop on CPU with a deterministic ROS clock."""
from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ftact import CAMERAS, STATE, WRENCH
from ftact_control import STATE_JOINT_NAMES
from validate_ftact_run import temporal_ensemble_checks
import numpy as np
import torch


def stamp(seconds):
    nanos = round(seconds*1e9)
    return NS(sec=nanos//10**9, nanosec=nanos % 10**9)


class Clock:
    def __init__(self): self.now, self.wall = 10., 0.
    def advance(self, seconds): self.now += seconds; self.wall += seconds
    def time_now(self): return NS(nanoseconds=round(self.now*1e9), to_msg=lambda: stamp(self.now))


class PolicyBase:
    def __init__(self, node): self._parent_node = node
    def time_now(self): return self._parent_node.clock.time_now()
    def get_logger(self): return self._parent_node.logger


def load_policy_module():
    def module(**values):
        result = ModuleType('stub'); result.__dict__.update(values); return result
    class Mode(NS): MODE_POSITION = 1
    stubs = {
        'rclpy': module(ok=lambda: True), 'rclpy.qos': module(qos_profile_sensor_data=None),
        'rclpy.callback_groups': module(ReentrantCallbackGroup=NS),
        'sensor_msgs': module(), 'sensor_msgs.msg': module(JointState=NS),
        'aic_model': module(), 'aic_model.policy': module(Policy=PolicyBase),
        'aic_control_interfaces': module(),
        'aic_control_interfaces.msg': module(MotionUpdate=NS, TrajectoryGenerationMode=Mode),
        'geometry_msgs': module(),
        'geometry_msgs.msg': module(Point=NS, Pose=NS, Quaternion=NS, Vector3=NS, Wrench=NS),
        'std_msgs': module(), 'std_msgs.msg': module(Header=NS),
    }
    spec = importlib.util.spec_from_file_location('temporal_policy_test', ROOT / 'policies/FTACT.py')
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(result)
    return result


class Model:
    def __init__(self, clock, first_delay):
        self.clock, self.first_delay, self.calls = clock, first_delay, 0
        self.policy = NS(reset=Mock())
    def predict_action_chunk(self, _batch):
        self.calls += 1
        self.clock.advance(self.first_delay if self.calls == 1 else .01)
        poses = torch.zeros(1, 20, 7)
        poses[0, :, 0] = self.calls*.001+torch.arange(20)*.0001
        poses[0, :, 6] = 1
        return poses


class PolicyIntegrationTests(unittest.TestCase):
    def run_policy(self, first_delay=0., tasks=1):
        module = load_policy_module()
        clock = Clock()
        node = NS(clock=clock, logger=NS(info=Mock(), error=Mock()), is_active=True,
                  goal_handle=NS(is_active=True, is_cancel_requested=False), _ftact_source_joints=deque(maxlen=500))
        model = Model(clock, first_delay)
        payload = {'format': 'aic-sfp0-act-v2', 'action_offset': 1, 'step': 3, 'smoke_test': True,
                   'config': {'chunk_size': 20}, 'execution': {'n_action_steps': 10},
                   'stats': {'action': {'min': [-1.]*7, 'max': [1.]*7}},
                   'task': {'target_module': 'nic_card_mount_0', 'port': 'sfp_port_0'}}
        arrays = {STATE: np.ones(6, dtype=np.float32), WRENCH: np.zeros(6, dtype=np.float32),
                  **{key: np.full((256, 288, 3), 100, dtype=np.uint8) for key in CAMERAS}}
        moves = []
        def observation():
            camera_stamp = stamp((np.floor((clock.now-10.)*20+1e-6))/20+9.9)
            header = NS(stamp=camera_stamp, frame_id='base_link')
            joints = NS(header=header, name=list(STATE_JOINT_NAMES), position=np.ones(6).tolist())
            node._ftact_source_joints.append(joints)
            return NS(center_image=NS(header=header), left_image=NS(header=header), right_image=NS(header=header),
                      joint_states=joints, wrist_wrench=NS(header=header),
                      controller_state=NS(header=header, tcp_pose=NS(position=NS(x=0., y=0., z=0.),
                                                      orientation=NS(x=0., y=0., z=0., w=1.))))
        with tempfile.TemporaryDirectory() as directory:
            settings = json.loads((ROOT / 'configs/sfp0_gazebo_v2.json').read_text())
            settings.update(device='cpu', amp='none', checkpoint='unused.pt', output_dir=directory,
                            max_task_seconds=1.5 if first_delay else .8)
            settings_path = Path(directory) / 'settings.json'
            settings_path.write_text(json.dumps(settings))
            with patch.dict(module.os.environ, FTACT_SETTINGS=str(settings_path)), \
                 patch.object(module, 'load_checkpoint', return_value=(model, payload)), \
                 patch.object(module, 'observation_arrays', return_value=arrays), \
                 patch.object(module, 'time', NS(monotonic=lambda: clock.wall, sleep=clock.advance)):
                # Existing checkpoints record 10-step chunk execution. The new
                # inference config overrides it to query every step without retraining.
                policy = module.FTACT(node)
                for i in range(tasks):
                    clock.now = 10.+i*10
                    task = NS(id=f'task_{i+1}', time_limit=2., target_module_name='nic_card_mount_0',
                              port_name='sfp_port_0', plug_type='sfp')
                    completed = policy.insert_cable(task, observation,
                        lambda **kwargs: moves.append(kwargs['motion_update']) or True, lambda _: None)
                    self.assertTrue(completed)
                policy.journal.close()
            rows = [json.loads(line) for line in (Path(directory) / 'inference.jsonl').read_text().splitlines()]
        node.logger.error.assert_not_called()
        return rows, model, moves

    def test_predicts_every_command_ensembles_overlap_and_resets_for_each_task(self):
        rows, model, moves = self.run_policy(tasks=2)
        predictions = [r for r in rows if r['type'] == 'prediction']
        self.assertGreater(len(predictions), 20)
        self.assertEqual(model.calls, len(predictions))
        self.assertEqual(len(moves), len(predictions))
        self.assertEqual(model.policy.reset.call_count, 2)
        self.assertEqual(rows[0]['execution_mode'], 'act_temporal_ensemble')
        self.assertEqual(rows[0]['replan_every_steps'], 1)
        self.assertTrue(all(temporal_ensemble_checks(predictions, .01).values()))
        for task_id in ('task_1', 'task_2'):
            task_rows = [r for r in predictions if r['task_id'] == task_id]
            self.assertEqual(task_rows[0]['ensemble_count'], 1)
            self.assertGreater(task_rows[-1]['ensemble_count'], 5)

    def test_stale_initial_prediction_is_not_put_into_the_ensemble(self):
        rows, model, moves = self.run_policy(first_delay=.7)
        predictions = [r for r in rows if r['type'] == 'prediction']
        self.assertEqual(len([r for r in rows if r['type'] == 'stale_prediction_dropped']), 1)
        self.assertEqual(model.calls, len(moves)+1)
        self.assertEqual(predictions[0]['ensemble_inference_ids'], [2])
        self.assertEqual(predictions[0]['ensemble_count'], 1)
        self.assertTrue(all(temporal_ensemble_checks(predictions, .01).values()))


if __name__ == '__main__':
    unittest.main()
