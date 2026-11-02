"""PACE context delivery/reset through the real ROS policy adapter on a fake clock."""
from collections import deque
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ftact import CAMERAS, STATE, WRENCH
from ftact_control import STATE_JOINT_NAMES
from pace_act import FORMAT, PHASE, HISTORY
from test_act_temporal_policy import Clock, Model, load_policy_module, stamp


class PACEPolicyTests(unittest.TestCase):
    def test_causal_context_reaches_model_before_ensembling_and_resets_between_tasks(self):
        module, clock = load_policy_module(), Clock()
        node = NS(clock=clock, logger=NS(info=Mock(), error=Mock()), is_active=True,
            goal_handle=NS(is_active=True, is_cancel_requested=False), _ftact_source_joints=deque(maxlen=500))
        class CaptureModel(Model):
            def predict_action_chunk(self, b):
                self.inputs.append({k: v.cpu().clone() for k, v in b.items() if 'images' not in k})
                return super().predict_action_chunk(b)
        model = CaptureModel(clock, 0.)
        model.inputs = []
        payload = {'format': FORMAT, 'step': 3, 'smoke_test': True, 'action_offset': 1,
            'config': {'chunk_size': 20, 'phase_alpha': .05, 'tactile_history': 4, 'correction_horizon': 5},
            'execution': {'n_action_steps': 1}, 'stats': {'action': {'min': [-1.] * 7, 'max': [1.] * 7}},
            'task': {'target_module': 'nic_card_mount_0', 'port': 'sfp_port_0'}}
        def observation():
            header = NS(stamp=stamp(np.floor((clock.now - 10) * 20 + 1e-6) / 20 + 9.9), frame_id='base_link')
            joints = NS(header=header, name=list(STATE_JOINT_NAMES), position=[1.] * 6)
            node._ftact_source_joints.append(joints)
            return NS(center_image=NS(header=header), joint_states=joints, wrist_wrench=NS(header=header),
                controller_state=NS(header=header, tcp_pose=NS(position=NS(x=0., y=0., z=0.),
                                                              orientation=NS(x=0., y=0., z=0., w=1.))))
        def arrays(*args):
            return {STATE: np.ones(6, dtype=np.float32),
                WRENCH: np.array([clock.now * .1, 0, 0, 0, 0, 0], dtype=np.float32),
                **{k: np.full((256, 288, 3), 100, dtype=np.uint8) for k in CAMERAS}}
        moves, boundaries = [], []
        with tempfile.TemporaryDirectory() as directory:
            settings = json.loads((ROOT / 'configs/sfp0_gazebo_v2.json').read_text())
            settings.update(device='cpu', amp='none', checkpoint='unused', output_dir=directory, max_task_seconds=.8)
            path = Path(directory) / 'settings.json'; path.write_text(json.dumps(settings))
            with patch.dict(module.os.environ, FTACT_SETTINGS=str(path)), \
                 patch.object(module, 'load_checkpoint', return_value=(model, payload)), \
                 patch.object(module, 'observation_arrays', side_effect=arrays), \
                 patch.object(module, 'time', NS(monotonic=lambda: clock.wall, sleep=clock.advance)):
                policy = module.FTACT(node)
                for i in range(2):
                    clock.now = 10. + 10 * i
                    boundaries.append(len(model.inputs))
                    task = NS(id=f'task_{i}', time_limit=2., target_module_name='nic_card_mount_0',
                              port_name='sfp_port_0', plug_type='sfp')
                    self.assertTrue(policy.insert_cable(task, observation,
                        lambda **kw: moves.append(kw['motion_update']) or True, lambda _: None))
                policy.journal.close()
            rows = [json.loads(line) for line in (Path(directory) / 'inference.jsonl').read_text().splitlines()]
        node.logger.error.assert_not_called()
        self.assertEqual(model.calls, len(moves))
        self.assertGreater(len(moves), 20)
        self.assertTrue(rows[0]['pace_phase_event'])
        self.assertEqual(rows[0]['execution_mode'], 'act_temporal_ensemble')
        for index in boundaries:
            b = model.inputs[index]
            torch.testing.assert_close(b[PHASE], b[WRENCH])
            torch.testing.assert_close(b[HISTORY], b[WRENCH][:, None].expand(-1, 4, -1))
        for b in model.inputs:
            torch.testing.assert_close(b[HISTORY][:, -1], b[WRENCH])
            self.assertLessEqual(b[PHASE][0, 0], b[WRENCH][0, 0])
        predictions = [r for r in rows if r['type'] == 'prediction']
        self.assertTrue(all(0 <= r['chunk_index'] < 20 for r in predictions))
        self.assertTrue(any(r['ensemble_count'] > 1 for r in predictions))


if __name__ == '__main__': unittest.main()
