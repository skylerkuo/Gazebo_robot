"""Run the real ROS policy loop with a controlled clock and generated chunks."""
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
sys.path.insert(0, str(ROOT/'scripts'))
from ftact import CAMERAS, STATE, WRENCH
from ftact_control import STATE_JOINT_NAMES
from validate_ftact_run import flow_receding_horizon_checks
from test_act_temporal_policy import Clock, Model, load_policy_module, stamp


class FlowPolicyTests(unittest.TestCase):
    def run_policy(self, first_delay=0., tasks=1, n_action_steps=1, method='flow'):
        module = load_policy_module()
        clock = Clock()
        node = NS(clock=clock, logger=NS(info=Mock(), error=Mock()), is_active=True,
            goal_handle=NS(is_active=True, is_cancel_requested=False), _ftact_source_joints=deque(maxlen=500))
        model = Model(clock, first_delay)
        model.configure_sampling = Mock()
        model.reset_sampler = Mock()
        payload = dict(format='aic-sfp0-flow-matching-v1', action_offset=1, step=3, smoke_test=True,
            config={'chunk_size':20}, execution={'n_action_steps':1},
            stats={'action':{'min':[-1.]*7,'max':[1.]*7}},
            task={'target_module':'nic_card_mount_0','port':'sfp_port_0'})
        if method == 'act':
            payload['format'] = 'aic-sfp0-act-v2'
            payload['execution']['n_action_steps'] = 10  # comparison config overrides old guidance
        arrays = {STATE:np.ones(6,dtype=np.float32), WRENCH:np.zeros(6,dtype=np.float32),
                  **{key:np.full((256,288,3),100,dtype=np.uint8) for key in CAMERAS}}
        def observation():
            header = NS(stamp=stamp(np.floor((clock.now-10)*20+1e-6)/20+9.9), frame_id='base_link')
            joints = NS(header=header, name=list(STATE_JOINT_NAMES), position=np.ones(6).tolist())
            node._ftact_source_joints.append(joints)
            return NS(center_image=NS(header=header), joint_states=joints, wrist_wrench=NS(header=header),
                controller_state=NS(header=header,tcp_pose=NS(position=NS(x=0.,y=0.,z=0.),
                                                           orientation=NS(x=0.,y=0.,z=0.,w=1.))))
        moves=[]
        with tempfile.TemporaryDirectory() as directory:
            config_name = 'sfp0_flow_gazebo.json' if method == 'flow' else 'sfp0_act_receding_gazebo.json'
            settings=json.loads((ROOT/'configs'/config_name).read_text())
            settings.update(device='cpu',amp='none',checkpoint='unused',output_dir=directory,
                            max_task_seconds=1.2,n_action_steps=n_action_steps)
            path=Path(directory)/'settings.json';path.write_text(json.dumps(settings))
            with patch.dict(module.os.environ, FTACT_SETTINGS=str(path)), \
                 patch.object(module,'load_checkpoint',return_value=(model,payload)), \
                 patch.object(module,'observation_arrays',return_value=arrays), \
                 patch.object(module,'time',NS(monotonic=lambda:clock.wall,sleep=clock.advance)):
                policy=module.FTACT(node)
                for i in range(tasks):
                    clock.now=10+i*10
                    task=NS(id=f'task_{i}',time_limit=2.,target_module_name='nic_card_mount_0',
                            port_name='sfp_port_0',plug_type='sfp')
                    self.assertTrue(policy.insert_cable(task,observation,
                        lambda **kwargs:moves.append(kwargs['motion_update']) or True,lambda _:None))
                policy.journal.close()
            rows=[json.loads(line) for line in (Path(directory)/'inference.jsonl').read_text().splitlines()]
        node.logger.error.assert_not_called()
        return rows,model,moves,settings

    def test_flow_replans_each_command_skips_latency_and_resets_seed_each_task(self):
        rows,model,moves,settings=self.run_policy(tasks=2)
        predictions=[r for r in rows if r['type']=='prediction']
        chunks=[r for r in rows if r['type']=='chunk_prediction']
        loaded=[r for r in rows if r['type']=='model_loaded']
        self.assertGreater(len(moves),20)
        self.assertEqual(len(chunks),len(moves))
        self.assertEqual(model.reset_sampler.call_count,2)
        self.assertTrue(all(r['chunk_index']>=1 for r in predictions))  # camera is 100ms behind now
        self.assertTrue(all('ensemble_count' not in r for r in predictions))
        checks=flow_receding_horizon_checks(predictions,chunks,loaded,dict(settings=settings,trials=['trial_1']))
        self.assertTrue(all(checks.values()),checks)
        chunks[0]['actions'][predictions[0]['chunk_index']][0]+=.1
        checks=flow_receding_horizon_checks(predictions,chunks,loaded,dict(settings=settings,trials=['trial_1']))
        self.assertFalse(checks['generated_samples_match_commands'])

    def test_stale_first_sample_is_dropped_and_multi_step_execution_is_available(self):
        rows,model,moves,settings=self.run_policy(first_delay=.6,n_action_steps=5)
        self.assertTrue(any(r['type']=='stale_prediction_dropped' for r in rows))
        chunks=[r for r in rows if r['type']=='chunk_prediction']
        self.assertTrue(chunks[0]['control_period_exceeded'])
        predictions=[r for r in rows if r['type']=='prediction']
        self.assertTrue(all(r['inference_id']!=1 for r in predictions))
        self.assertTrue(any(not r['new_chunk'] for r in predictions))
        self.assertTrue(all(r['observation_age_seconds']<=settings['max_observation_age_seconds'] for r in predictions))

    def test_optional_act_baseline_uses_the_same_replanning_and_timing_without_ensembling(self):
        rows,model,moves,settings=self.run_policy(method='act')
        loaded=next(r for r in rows if r['type']=='model_loaded')
        predictions=[r for r in rows if r['type']=='prediction']
        self.assertEqual(loaded['execution_mode'],'act_receding_horizon')
        self.assertEqual(loaded['replan_every_steps'],1)
        self.assertTrue(all(r['new_chunk'] and r['chunk_index']>=1 for r in predictions))
        self.assertEqual(model.calls,len(moves))
        self.assertTrue(all('ensemble_count' not in r for r in predictions))
        model.configure_sampling.assert_not_called()
        model.reset_sampler.assert_not_called()


if __name__=='__main__':
    unittest.main()
