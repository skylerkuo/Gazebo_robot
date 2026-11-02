"""Automatic handoff uses simulation time, records no fake human actions, and settles."""
from pathlib import Path
import sys
from unittest.mock import patch
import importlib.util
from types import ModuleType,SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from automatic_expert_control import AutomaticExpertOperator,InPlaceExpertAlignment
from human_control import expert_anchor_frames
from train_sfp0_act_multiview import parse_args


def test_handoff_waits_for_model_then_runs_expert_once_and_finishes():
    op=AutomaticExpertOperator(20)
    assert op.poll({})['mode']=='model'
    assert op.poll({'elapsed_seconds':19.95})['mode']=='model'
    state=op.poll({'elapsed_seconds':20})
    assert state['mode']=='human' and state['correction_active'] and not any(state['axes'])
    version=state['version']
    assert op.poll({'elapsed_seconds':21})['version']==version
    assert op.poll({'elapsed_seconds':79.99})['mode']=='human'
    state=op.poll({'elapsed_seconds':80})
    assert state['mode']=='hold' and state['takeover_reason']=='expert_time_budget_exhausted'
    assert op.poll({'elapsed_seconds':82})['finish']


def test_insertion_event_holds_without_expert_and_force_limit_discards():
    op=AutomaticExpertOperator(20)
    state=op.poll({'elapsed_seconds':10,'insertion_event':True})
    assert state['mode']=='hold' and not state['finish'] and op.expert_started is None
    assert op.poll({'elapsed_seconds':12,'insertion_event':True})['finish']
    state=AutomaticExpertOperator(20).poll({'request_hold':True})
    assert state['discard'] and state['finish'] and state['mode']=='hold'


def test_expert_labels_are_separate_from_human_and_never_include_model_prefix():
    sources=['model']*10+['expert']*30+['hold']*5
    times=np.arange(45)*.05;segments=[1]*10+[2]*30+[3]*5
    active=[False]*10+[True]*30+[False]*5
    assert expert_anchor_frames(sources,segments,active,times)==[]
    assert expert_anchor_frames(sources,segments,active,times,source='expert')==list(range(9,20))


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_handoff_rejected(value):
    with pytest.raises(ValueError):AutomaticExpertOperator(value)


def test_expert_training_config_refreshes_data_and_evaluates_new_1500_val():
    path=Path(__file__).resolve().parents[1]/'configs/sfp0_act_expert_finetune.json'
    with patch.object(sys,'argv',['train','--config',str(path),'--device','cpu','--amp','none']):
        _,cfg,resume=parse_args()
    assert resume is None and cfg.init_from and cfg.expert_only and cfg.dataset_root.endswith('aic_sfp0_expert_v1')
    assert cfg.eval_samples==1500 and cfg.eval_every==500 and cfg.learning_rate==3e-5


def test_in_place_alignment_never_raises_and_waits_for_tip_alignment():
    actual=np.array([0.,0.,.2,0,0,0,1]);port=actual.copy();plug=actual.copy();plug[0]=.004
    official=actual.copy();official[2]=.4
    controller=InPlaceExpertAlignment(actual,0)
    for now in np.arange(0,1,.05):
        target,info=controller.target(official,actual,plug,port,now)
        assert target[2]==.2 and info['phase']=='align'
    plug[0]=0
    for now in np.arange(1,1.3,.05):
        target,info=controller.target(official,actual,plug,port,now)
        assert target[2]<=.2
    target,info=controller.target(official,actual,plug,port,1.4)
    assert info['phase']=='insert' and .199<=target[2]<.2
    height=target[2]
    # A lagging impedance servo must not pin the reference to actual Z - 1 mm.
    for now in np.arange(1.45,2,.05):target,_=controller.target(official,actual,plug,port,now)
    assert target[2]<.199 and target[2]>=controller.floor
    plug[0]=.005
    target,info=controller.target(official,actual,plug,port,2.05)
    assert info['phase']=='align' and target[2]<=height
    # Unexpected downward displacement must not trigger an upward recovery.
    actual[2]=.16
    target,_=controller.target(official,actual,plug,port,2.1)
    assert target[2]==.16


def test_in_place_alignment_requires_rotation_and_caps_descent():
    pose=np.array([0.,0.,.2,0,0,0,1]);plug=pose.copy();port=pose.copy();port[2]=0
    plug[5:]=[np.sin(np.deg2rad(3)),np.cos(np.deg2rad(3))]
    controller=InPlaceExpertAlignment(pose,0)
    target,info=controller.target(pose,pose,plug,port,1)
    assert info['phase']=='align' and target[2]==pose[2]
    actual=pose.copy()
    for now in np.arange(1.05,45,.05):
        target,info=controller.target(pose,actual,pose,port,now)
        assert target[2]<=actual[2]+1e-12
        actual=target.copy()
    assert target[2]==pytest.approx(.12) and info['descent_limit_reached']


def test_stalled_servo_reference_advances_past_old_clamp_and_stops_at_tip_depth():
    actual=np.array([0.,0.,.237,0,0,0,1]);plug=actual.copy();plug[2]=.04
    port=plug.copy();port[2]=0
    ctl=InPlaceExpertAlignment(actual,0)
    heights=[]
    for now in np.arange(0,30,.05):
        target,info=ctl.target(actual,actual,plug,port,now,force_n=20.)
        heights.append(target[2])
    assert np.max(np.diff(heights))<=1e-12
    assert ctl.floor==pytest.approx(.187)
    assert target[2]==pytest.approx(.187) and info['target_lead_mm']==pytest.approx(50)
    assert info['stalled'] and info['descent_limit_reached']


def test_force_pause_and_alignment_hysteresis_do_not_raise_reference():
    pose=np.array([0.,0.,.23,0,0,0,1]);plug=pose.copy();plug[2]=.04
    port=plug.copy();port[2]=0;ctl=InPlaceExpertAlignment(pose,0)
    for now in np.arange(0,1,.05):target,info=ctl.target(pose,pose,plug,port,now,force_n=20.)
    assert info['phase']=='insert'
    height=target[2]
    plug[0]=.002
    target,info=ctl.target(pose,pose,plug,port,1,force_n=20.)
    assert info['phase']=='insert' and target[2]<height
    height=target[2]
    target,info=ctl.target(pose,pose,plug,port,1.05,force_n=36.)
    assert info['phase']=='contact_pause' and info['force_paused'] and target[2]==height
    target,info=ctl.target(pose,pose,plug,port,1.1,force_n=30.)
    assert info['force_paused'] and target[2]==height
    plug[0]=0
    for now in np.arange(1.15,1.7,.05):target,info=ctl.target(pose,pose,plug,port,now,force_n=25.)
    assert info['phase']=='insert' and not info['force_paused'] and target[2]<height
    height=target[2];plug[0]=.004
    target,info=ctl.target(pose,pose,plug,port,1.75,force_n=25.)
    assert info['phase']=='align' and target[2]==height


def test_expert_force_ceiling_and_custom_time_budget():
    op=AutomaticExpertOperator(20,45)
    assert op.poll({'elapsed_seconds':20,'force_n':20})['mode']=='human'
    assert op.poll({'elapsed_seconds':30,'force_n':40})['mode']=='human'
    state=op.poll({'elapsed_seconds':30.05,'force_n':46})
    assert state['finish'] and state['discard'] and state['takeover_reason']=='expert_contact_force_limit'
    op=AutomaticExpertOperator(20,45);op.poll({'elapsed_seconds':20})
    assert op.poll({'elapsed_seconds':64.9})['mode']=='human'
    assert op.poll({'elapsed_seconds':65})['mode']=='hold'
    assert op.poll({'elapsed_seconds':67})['finish']
    for invalid in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):AutomaticExpertOperator(20,invalid)


def test_axial_impedance_changes_only_during_expert_takeover():
    # Import the real subclass without ROS, while preserving the base motion contract.
    class Base:
        def insert_cable(self,task,*args):
            self.task_starts.append((self.operator.mode,self.operator.expert_started,self.operator.inserted))
            self.operator.poll({'elapsed_seconds':20,'force_n':20})
            self.operator.poll({'elapsed_seconds':21,'insertion_event':True})
            return True
        def motion(self,command):
            return NS(pose=command,target_stiffness=np.diag(self.settings['stiffness']).flatten().tolist(),
                target_damping=np.diag(self.settings['damping']).flatten().tolist())
    human=ModuleType('policies.HumanFTACT');human.HumanFTACT=Base
    cheat=ModuleType('aic_example_policies.ros.CheatCode');cheat.CheatCode=object
    ros=ModuleType('rclpy.time');ros.Time=object
    spec=importlib.util.spec_from_file_location('auto_expert_motion_test',Path(__file__).resolve().parents[1]/'policies/AutoExpertFTACT.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'policies.HumanFTACT':human,'aic_example_policies.ros.CheatCode':cheat,'rclpy.time':ros}):
        spec.loader.exec_module(module)
    policy=module.AutoExpertFTACT();policy.operator=NS(mode='model')
    policy.settings={'stiffness':[90,90,90,50,50,50],'damping':[50,50,50,20,20,20]}
    command=np.array([0,0,.2,0,0,0,1.])
    model_motion=policy.motion(command)
    policy.operator.mode='human';expert_motion=policy.motion(command)
    policy.operator.mode='hold';hold_motion=policy.motion(command)
    np.testing.assert_array_equal(model_motion.target_stiffness,hold_motion.target_stiffness)
    model_k=np.array(model_motion.target_stiffness).reshape(6,6)
    expert_k=np.array(expert_motion.target_stiffness).reshape(6,6)
    assert expert_k[2,2]>model_k[2,2]
    expert_k[2,2]=model_k[2,2]
    np.testing.assert_array_equal(expert_k,model_k)
    np.testing.assert_array_equal(expert_motion.pose,command)
    # A second task on the SAME loaded model must begin with a fresh operator,
    # rather than inheriting the preceding task's insertion/hold state.
    module.CheatCode=lambda node:NS()
    policy._parent_node=NS();policy.task_starts=[];policy.settings['expert_after_seconds']=20.
    for i in range(2):assert policy.insert_cable(NS(id=f'episode_{i}'),None,None,None)
    assert policy.task_starts==[('model',None,None),('model',None,None)]
