"""Exercise four-view ACT's actual policy loop, synchronization waits and resets."""
from collections import deque
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import Mock,patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from multiview_control import CAMERA_KEYS
from ftact_control import STATE_JOINT_NAMES
from validate_ftact_run import temporal_ensemble_checks
from test_act_temporal_policy import Clock,Model,load_policy_module,stamp


@pytest.mark.parametrize('checkpoint_format', ['aic-sfp0-act-multiview-v1',
    'aic-sfp0-act-pace-multiview-v1', 'aic-sfp0-act-pace-pred-multiview-v1',
    'aic-sfp0-flow-matching-multiview-v1','aic-sfp0-supervised-residual-flow-multiview-v1',
    'aic-sfp0-stacked-residual-flow-multiview-v1',
    'aic-sfp0-alternating-act-residual-flow-multiview-v1'])
def test_four_views_reach_model_every_step_with_sync_retries_and_task_reset(tmp_path, checkpoint_format):
    module=load_policy_module();clock=Clock()
    subscriptions={}
    def subscribe(kind,topic,callback,*args,**kwargs):
        # rclpy inspects the Python callback; deque.append has no signature.
        inspect.signature(callback)
        subscriptions[topic]=callback
        return NS()
    node=NS(clock=clock,logger=NS(info=Mock(),error=Mock()),is_active=True,
        goal_handle=NS(is_active=True,is_cancel_requested=False),
        _ftact_source_joints=deque(maxlen=500),create_subscription=subscribe)
    class FourViewModel(Model):
        def configure_sampling(self,**kwargs):self.sampler_settings=kwargs
        def reset_sampler(self,seed):self.sampler_seed=seed
        def predict_action_chunk(self,batch):
            assert set(CAMERA_KEYS)<=batch.keys()
            assert all(batch[key].shape==(1,3,256,288) for key in CAMERA_KEYS)
            assert batch['observation.environment_state'].shape==(1,6)
            if 'pace' in checkpoint_format:
                from pace_act import PHASE,HISTORY
                assert batch[PHASE].shape==(1,6) and batch[HISTORY].shape==(1,4,6)
                assert not any(k.startswith('supervision.') for k in batch)
            return super().predict_action_chunk(batch)
    model=FourViewModel(clock,0.)
    profile=json.loads((ROOT/'configs/sfp0_sideview.json').read_text())
    profile.update(record_width=5,record_height=4)
    info=NS(width=5,height=4,k=[5.,0,2.5,0,5.,2,0,0,1.])
    expected={name:{'width':5,'height':4,'k':info.k} for name in ('left','center','right','side')}
    payload={'format':checkpoint_format,'action_offset':1,'step':2500,'smoke_test':False,
        'config':{'chunk_size':20},'execution':{'n_action_steps':1},'camera_order':list(CAMERA_KEYS),
        'stats':{'action':{'min':[-1.]*7,'max':[1.]*7}},
        'task':{'target_module':'nic_card_mount_0','port':'sfp_port_0'}}
    if 'pace' in checkpoint_format:
        payload['config'].update(phase_alpha=.05,tactile_history=4,correction_horizon=5)
    settings=json.loads((ROOT/'configs/sfp0_multiview_gazebo.json').read_text())
    settings.update(device='cpu',amp='none',checkpoint='unused',output_dir=str(tmp_path),max_task_seconds=.8,
        collection_settings=profile,expected_camera_info=expected,camera_order=list(CAMERA_KEYS))
    is_flow='flow-matching' in checkpoint_format or 'residual-flow' in checkpoint_format
    if is_flow:
        settings.update(execution_mode='sfp0_flow_receding_horizon',ode_steps=10,ode_solver='euler',noise_seed=42)
    (tmp_path/'settings.json').write_text(json.dumps(settings))
    sensors=ModuleType('sensor_msgs.msg');sensors.Image=NS;sensors.CameraInfo=NS
    package=ModuleType('sensor_msgs');package.msg=sensors
    seen={}
    def observation():
        t=np.floor((clock.now-10.)*20+1e-6)/20+9.9
        header=NS(stamp=stamp(t),frame_id='base_link')
        image=NS(header=header,width=5,height=4,step=15,encoding='rgb8',data=np.arange(60,dtype=np.uint8).tobytes())
        joints=NS(header=header,name=list(STATE_JOINT_NAMES),position=[1.,2,3,4,5,6])
        node._ftact_source_joints.append(joints)
        subscriptions['/side_camera/camera_info'](info)
        seen[t]=seen.get(t,0)+1
        # Delay the matching side frame once per wrist stamp, also mid-task.
        if seen[t]>=2:subscriptions['/side_camera/image'](image)
        wrench=NS(header=header,wrench=NS(force=NS(x=1.,y=2.,z=3.),torque=NS(x=0.,y=0.,z=0.)))
        return NS(left_image=image,center_image=image,right_image=image,wrist_wrench=wrench,joint_states=joints,
            left_camera_info=info,center_camera_info=info,right_camera_info=info,
            controller_state=NS(header=header,tcp_pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))))
    moves=[]
    with patch.dict(sys.modules,{'sensor_msgs':package,'sensor_msgs.msg':sensors}), \
         patch.dict(module.os.environ,FTACT_SETTINGS=str(tmp_path/'settings.json')), \
         patch.object(module,'load_checkpoint',return_value=(model,payload)), \
         patch.object(module,'time',NS(monotonic=lambda:clock.wall,sleep=clock.advance)):
        policy=module.FTACT(node)
        for i in range(2):
            clock.now=10.+i*10
            task=NS(id=f'task_{i}',time_limit=2.,target_module_name='nic_card_mount_0',port_name='sfp_port_0',plug_type='sfp')
            assert policy.insert_cable(task,observation,lambda **kw:moves.append(kw['motion_update']) or True,lambda _:None)
        policy.journal.close()
    rows=[json.loads(line) for line in (tmp_path/'inference.jsonl').read_text().splitlines()]
    predictions=[r for r in rows if r['type']=='prediction']
    assert len(moves)==model.calls==len(predictions)>10
    assert model.policy.reset.call_count==2 and not node.logger.error.called
    if not is_flow:assert all(temporal_ensemble_checks(predictions,.01).values())
    else:
        assert model.sampler_seed==42 and model.sampler_settings['steps']==10
        assert all('ensemble_count' not in r for r in predictions)
        assert all(0<=r['chunk_index']<20 for r in predictions)
    assert all(r['camera_max_age_seconds']==0 and len(r['camera_stamps'])==4 for r in predictions)
    for task in ('task_0','task_1'):
        if not is_flow:assert next(r for r in predictions if r['task_id']==task)['ensemble_count']==1
    assert len([r for r in rows if r['type']=='waiting_for_initial_observation'])>=1
