"""Four-view sensor causality, dataset preprocessing, checkpoint/scene contracts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from ftact_control import STATE_JOINT_NAMES
from multiview_control import CAMERA_KEYS,multiview_arrays,ObservationNotReady
from act_multiview import FORMAT,index_digest
from run_sfp0_multiview_gazebo import collection_contract,snapshot_checkpoint,inference_scene


def stamp(t):
    n=round(t*1e9)
    return NS(sec=n//10**9,nanosec=n%10**9)


def sensors():
    profile=json.loads((ROOT/'configs/sfp0_sideview.json').read_text())
    profile.update(record_width=5,record_height=4)
    info=NS(width=5,height=4,k=[5.,0,2.5,0,5.,2,0,0,1.])
    rgb=np.arange(60,dtype=np.uint8).reshape(4,5,3)
    def image(t=1.,offset=0):
        return NS(header=NS(stamp=stamp(t)),height=4,width=5,step=15,encoding='rgb8',data=(rgb+offset).tobytes())
    joints=NS(header=NS(stamp=stamp(.998)),name=list(STATE_JOINT_NAMES),position=[1.,2,3,4,5,6])
    force=NS(header=NS(stamp=stamp(.998)),wrench=NS(force=NS(x=1.,y=2.,z=3.),torque=NS(x=4.,y=5.,z=6.)))
    observation=NS(left_image=image(),center_image=image(),right_image=image(),joint_states=joints,wrist_wrench=force,
        left_camera_info=info,center_camera_info=info,right_camera_info=info)
    expected={name:{'width':5,'height':4,'k':list(info.k)} for name in ('left','center','right','side')}
    return profile,info,expected,observation,joints,image


def test_four_view_inputs_choose_causal_side_and_joints_and_match_training_resize():
    profile,info,expected,obs,joints,image=sensors()
    future=copy.deepcopy(joints);future.header.stamp=stamp(1.001);future.position=[99.]*6
    arrays,chosen,details=multiview_arrays(obs,[image(1.,20),image(1.01,100)],[joints,future],info,profile,expected)
    assert chosen is joints and set(CAMERA_KEYS)<=arrays.keys()
    np.testing.assert_array_equal(arrays['observation.state'],joints.position)
    np.testing.assert_array_equal(arrays['observation.environment_state'],[1,2,3,4,5,6])
    rgb=np.frombuffer(image(1.,20).data,np.uint8).reshape(4,5,3)
    np.testing.assert_array_equal(arrays[CAMERA_KEYS[-1]],cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA))
    assert details['camera_stamps']['side']==1. and details['camera_max_age_seconds']==0
    assert details['joint_age_seconds']==pytest.approx(.002)


@pytest.mark.parametrize('problem',['no_side','future_side','stale_side','future_joint','stale_joint',
                                    'future_force','stale_force','future_wrist','no_info'])
def test_missing_future_or_stale_sensors_retry_without_reusing_wrong_data(problem):
    profile,info,expected,obs,joints,image=sensors()
    side=[image()];joint_list=[joints]
    if problem=='no_side':side=[]
    if problem=='future_side':side=[image(1.01)]
    if problem=='stale_side':side=[image(.95)]
    if problem=='future_joint':joints.header.stamp=stamp(1.001)
    if problem=='stale_joint':joints.header.stamp=stamp(.8)
    if problem=='future_force':obs.wrist_wrench.header.stamp=stamp(1.01)
    if problem=='stale_force':obs.wrist_wrench.header.stamp=stamp(.8)
    if problem=='future_wrist':obs.left_image.header.stamp=stamp(1.01)
    if problem=='no_info':info=None
    with pytest.raises(ObservationNotReady):multiview_arrays(obs,side,joint_list,info,profile,expected)


def test_wrong_camera_calibration_or_resolution_is_rejected():
    profile,info,expected,obs,joints,image=sensors()
    expected['side']['k'][0]=9.
    with pytest.raises(ValueError,match='calibration'):
        multiview_arrays(obs,[image()],[joints],info,profile,expected)
    expected['side']['k'][0]=5.
    obs.left_image.width=99
    with pytest.raises(ValueError,match='resolution'):
        multiview_arrays(obs,[image()],[joints],info,profile,expected)


def test_checkpoint_binds_scene_profile_to_frozen_training_sources(tmp_path):
    profile=json.loads((ROOT/'configs/sfp0_sideview.json').read_text())
    (tmp_path/'_collection').mkdir();(tmp_path/'run/raw').mkdir(parents=True)
    calibration={'side_camera':copy.deepcopy(profile['camera']),'record_width':576,'record_height':512,
        'source_camera_info':{name:{'width':576,'height':512,'k':[1.]*9} for name in ('left','center','right','side')}}
    (tmp_path/'run/raw/episode_000003_calibration.json').write_text(json.dumps(calibration))
    entry={'accepted':True,'episode_index':0,'attempt':3,'run':'run','validation':{'parquet_sha256':'source'}}
    manifest={'task':{'plug':'sfp_tip','target_module':'nic_card_mount_0','port':'sfp_port_0'},'settings':profile,'attempts':[entry]}
    (tmp_path/'_collection/manifest.json').write_text(json.dumps(manifest))
    index={'episodes':[{'episode_index':0,'source_attempt':3,'source_sha256':'source'}]}
    index['index_content_sha256']=index_digest(index)
    (tmp_path/'index.json').write_text(json.dumps(index))
    payload={'format':FORMAT,'camera_order':list(CAMERA_KEYS),'action_offset':1,'task':manifest['task'],
        'config':{'chunk_size':20,'dataset_root':str(tmp_path),'index_path':str(tmp_path/'index.json')},
        'data_index_sha256':index['index_content_sha256']}
    result,info,provenance=collection_contract(payload)
    assert result==profile and info==calibration['source_camera_info']
    assert provenance['data_index_sha256']==index['index_content_sha256']
    profile['camera']['position_world'][0]+=1.
    (tmp_path/'_collection/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='geometry/resolution'):collection_contract(payload)
    profile['camera']=copy.deepcopy(calibration['side_camera'])
    (tmp_path/'_collection/manifest.json').write_text(json.dumps(manifest))
    payload['data_index_sha256']='changed'
    with pytest.raises(ValueError,match='checksum'):collection_contract(payload)
    payload['data_index_sha256']=index['index_content_sha256'];entry['validation']['parquet_sha256']='changed'
    (tmp_path/'_collection/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='changed since training'):collection_contract(payload)
    payload['camera_order']=payload['camera_order'][:3]
    with pytest.raises(ValueError,match='four-view'):collection_contract(payload)


def test_checkpoint_snapshot_is_independent_of_later_atomic_best_updates(tmp_path):
    source=tmp_path/'best.pt';snapshot=tmp_path/'checkpoint.pt'
    original=b'complete checkpoint';source.write_bytes(original)
    digest=snapshot_checkpoint(source,snapshot)
    replacement=tmp_path/'next.pt';replacement.write_bytes(b'new training checkpoint');replacement.replace(source)
    assert snapshot.read_bytes()==original
    import hashlib
    assert digest==hashlib.sha256(original).hexdigest()


def test_expert_snapshots_share_immutable_bytes_and_survive_source_updates(tmp_path):
    import os
    source=tmp_path/'best.pt';source.write_bytes(b'original checkpoint')
    cache=tmp_path/'shared';first=tmp_path/'first.pt';second=tmp_path/'second.pt'
    digest=snapshot_checkpoint(source,first,shared_cache=cache)
    assert snapshot_checkpoint(source,second,shared_cache=cache)==digest
    assert os.path.samefile(first,second)
    replacement=tmp_path/'next.pt';replacement.write_bytes(b'new checkpoint');replacement.replace(source)
    third=tmp_path/'third.pt';assert snapshot_checkpoint(source,third,shared_cache=cache)!=digest
    assert first.read_bytes()==second.read_bytes()==b'original checkpoint'
    assert third.read_bytes()==b'new checkpoint'


def test_portable_camera_contract_survives_old_dataset_deletion_and_checks_binding(tmp_path):
    profile,_,info,_,_,_=sensors()
    payload={'format':FORMAT,'camera_order':list(CAMERA_KEYS),'action_offset':1,
        'config':{'chunk_size':20},'task':{'plug':'sfp_tip','target_module':'nic_card_mount_0','port':'sfp_port_0'},
        'data_index_sha256':'old-index'}
    body={'format':'aic-sfp0-portable-collection-contract-v1','data_index_sha256':'old-index',
        'profile':profile,'camera_info':info,'source_contract':{'collection_root':'/deleted/data'}}
    body['contract_sha256']=index_digest(body);p=tmp_path/'contract.json';p.write_text(json.dumps(body))
    result,expected,source=collection_contract(payload,p)
    assert result==profile and expected==info and source['collection_root']=='/deleted/data'
    payload['data_index_sha256']='other'
    with pytest.raises(ValueError,match='binding'):collection_contract(payload,p)
    payload['data_index_sha256']='old-index';body['profile']['camera']['near']=999;p.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='checksum'):collection_contract(payload,p)


def test_default_scene_varies_within_training_bounds_and_is_reproducible(monkeypatch):
    import yaml
    profile=json.loads((ROOT/'configs/sfp0_sideview.json').read_text())
    source=yaml.safe_load((ROOT/'runtime/rootfs/ws_aic/install/share/aic_engine/config/sample_config.yaml').read_text())
    original_source=copy.deepcopy(source);original_profile=copy.deepcopy(profile)
    draws=iter([17,29])
    monkeypatch.setattr('run_sfp0_multiview_gazebo.secrets.randbelow',lambda _:next(draws))
    a,scene_a=inference_scene(source,profile,60)
    b,scene_b=inference_scene(source,profile,60)
    assert scene_a['mode']==scene_b['mode']=='random'
    assert scene_a['scene_attempt']==18 and scene_b['scene_attempt']==30
    assert scene_a['task_board_pose']!=scene_b['task_board_pose']
    for engine,scene in ((a,scene_a),(b,scene_b)):
        assert list(engine['trials'])==['trial_1']
        task=engine['trials']['trial_1']['tasks']['task_1']
        assert (task['target_module_name'],task['port_name'],task['plug_name'])==('nic_card_mount_0','sfp_port_0','sfp_tip')
        assert task['time_limit']==65
        for axis in ('x','y','z'):
            assert abs(scene['translation_offset_m'][axis])<=profile['variation'][f'board_{axis}_m']+1e-12
        assert abs(scene['yaw_offset_rad'])<=profile['variation']['board_yaw_rad']+1e-12
    replay,replayed=inference_scene(source,profile,60,scene_a['scene_attempt'])
    assert replay==a and replayed['mode']=='fixed'
    assert replayed['task_board_pose']==scene_a['task_board_pose']
    assert source==original_source and profile==original_profile


def test_explicit_zero_preserves_official_sample_placement():
    import yaml
    profile=json.loads((ROOT/'configs/sfp0_sideview.json').read_text())
    source=yaml.safe_load((ROOT/'runtime/rootfs/ws_aic/install/share/aic_engine/config/sample_config.yaml').read_text())
    engine,scene=inference_scene(source,profile,60,0)
    assert scene['mode']=='fixed' and scene['scene_attempt']==0
    assert scene['translation_offset_m']==dict(x=0.,y=0.,z=0.) and scene['yaw_offset_rad']==0.
    assert engine['trials']['trial_1']['scene']==source['trials']['trial_1']['scene']
    with pytest.raises(ValueError,match='nonnegative'):inference_scene(source,profile,60,-1)
