"""The real takeover loop suppresses stale model commands and records the applied source."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import Mock,patch
from collections import deque

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from test_act_temporal_policy import Clock,Model,load_policy_module,stamp
from ftact_control import STATE_JOINT_NAMES
from multiview_control import CAMERA_KEYS
from validate_ftact_run import temporal_ensemble_checks


def test_operator_takeover_during_inference_suppresses_model_output_and_resets_chunks(tmp_path):
    base=load_policy_module();clock=Clock()
    node=NS(clock=clock,logger=NS(info=Mock(),error=Mock()),is_active=True,
        goal_handle=NS(is_active=True,is_cancel_requested=False),_ftact_source_joints=deque(maxlen=500),
        create_subscription=lambda *a,**k:NS())
    commands=[]
    class Recorder:
        def __init__(self,node,settings):
            self.output=Path(settings['raw_dir']);self.output.mkdir(parents=True)
            self.active=NS(frames=0);self.insertion_events=[]
        def begin(self,task):pass
        def command(self,motion,**provenance):commands.append((motion,provenance))
        def finish(self,completed,**kwargs):return {'policy_return':completed,'writer_error':None}
    class Link:
        def __init__(self,_):pass
        def poll(self,status):
            json.dumps(status,allow_nan=False)  # Includes numpy-derived force-limit comparisons.
            t=clock.now-10
            phase=0 if t<.15 else 1 if t<.6 else 2 if t<.8 else 3
            return {'mode':('model','human','hold','model')[phase],'version':phase,
                'axes':[0,0,1 if phase==1 else 0,0,0,0],
                'translation_step_m':.0005,'rotation_step_rad':.01,'finish':t>=1.25,'discard':False}
        def close(self):pass
    policy_module=ModuleType('policies.FTACT');policy_module.FTACT=base.FTACT
    recorder_module=ModuleType('sideview_recorder');recorder_module.Recorder=Recorder
    sensors=ModuleType('sensor_msgs.msg');sensors.Image=NS;sensors.CameraInfo=NS
    package=ModuleType('sensor_msgs');package.msg=sensors
    profile=json.loads((ROOT/'configs/sfp0_sideview.json').read_text());profile.update(record_width=5,record_height=4)
    info=NS(width=5,height=4,k=[5.,0,2.5,0,5.,2,0,0,1.])
    expected={n:{'width':5,'height':4,'k':info.k} for n in ('left','center','right','side')}
    settings=json.loads((ROOT/'configs/sfp0_multiview_gazebo.json').read_text())
    settings.update(device='cpu',amp='none',checkpoint='unused',output_dir=str(tmp_path),raw_dir=str(tmp_path/'raw'),
        camera_order=list(CAMERA_KEYS),collection_settings=profile,expected_camera_info=expected,
        max_task_seconds=2.,human_operator={})
    path=tmp_path/'settings.json';path.write_text(json.dumps(settings))
    payload={'format':'aic-sfp0-act-multiview-v1','camera_order':list(CAMERA_KEYS),'action_offset':1,'step':1,
        'smoke_test':True,'config':{'chunk_size':20},'execution':{'n_action_steps':1},
        'stats':{'action':{'min':[-1.]*7,'max':[1.]*7}},
        'task':{'target_module':'nic_card_mount_0','port':'sfp_port_0'}}
    model=Model(clock,.2)
    spec=importlib.util.spec_from_file_location('human_policy_test',ROOT/'policies/HumanFTACT.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'policies':ModuleType('policies'),'policies.FTACT':policy_module,
            'sideview_recorder':recorder_module,'sensor_msgs':package,'sensor_msgs.msg':sensors}), \
         patch.dict(base.os.environ,FTACT_SETTINGS=str(path)),patch.object(base,'load_checkpoint',return_value=(model,payload)):
        spec.loader.exec_module(module)
        with patch.object(module,'OperatorLink',Link),patch.object(module,'time',NS(monotonic=lambda:clock.wall,sleep=clock.advance)):
            policy=module.HumanFTACT(node)
            def observation():
                t=np.floor((clock.now-10)*20+1e-6)/20+9.9
                h=NS(stamp=stamp(t),frame_id='base_link')
                image=NS(header=h,width=5,height=4,step=15,encoding='rgb8',data=np.arange(60,dtype=np.uint8).tobytes())
                joints=NS(header=h,name=list(STATE_JOINT_NAMES),position=[1.,2,3,4,5,6])
                node._ftact_source_joints.append(joints);node._ftact_side_images.append(image);node._ftact_side_info=info
                policy.recorder.active.frames+=1
                return NS(left_image=image,center_image=image,right_image=image,left_camera_info=info,
                    center_camera_info=info,right_camera_info=info,joint_states=joints,
                    wrist_wrench=NS(header=h,wrench=NS(force=NS(x=1.,y=2.,z=3.),torque=NS(x=0.,y=0.,z=0.))),
                    controller_state=NS(header=h,tcp_pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))))
            task=NS(id='task_1',target_module_name='nic_card_mount_0',port_name='sfp_port_0',plug_type='sfp',time_limit=3)
            assert policy.insert_cable(task,observation,lambda **kw:True,lambda _:None)
            policy.journal.close()
    rows=[json.loads(l) for l in (tmp_path/'inference.jsonl').read_text().splitlines()]
    predictions=[r for r in rows if r['type']=='prediction']
    assert predictions[0]['control_source']=='human'  # The first in-flight model prediction was preempted.
    assert set(r['control_source'] for r in predictions)=={'human','hold','model'}
    model_rows=[r for r in predictions if r['control_source']=='model']
    assert model_rows[0]['ensemble_count']==1
    assert all(temporal_ensemble_checks(model_rows,.01).values())
    assert all(p['source'] in ('model','human','hold') for _,p in commands)
    assert all(p['source']=='human' for _,p in commands if p.get('human_active',False))
    assert not node.logger.error.called
