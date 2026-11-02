"""New data membership, train-only normalization and frozen resume for ACT + Flow."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from act_multiview import file_digest,index_digest,read_index,split_episodes
from alternating_residual_flow_multiview import AlternatingResidualFlowConfig
from ftact import STATE,WRENCH
from human_control import expert_anchor_frames
from human_dataset import EXPERT_FORMAT
from prepare_expert_only import prepare_expert_only
from rollout_sampling import rollout_windows
from sideview_data import TASK
import train_sfp0_alternating_residual_flow_multiview as trainer


def collection(tmp_path,monkeypatch):
    root=tmp_path/'dataset';root.mkdir();(root/'_collection').mkdir()
    monkeypatch.setattr('prepare_expert_only.DATA_DRIVE_ROOT',tmp_path)
    monkeypatch.setattr('act_multiview.SUPPORTED_VALIDATION_SAMPLES',(10,1000,1500,2000))
    _,val=split_episodes(list(range(6)),43)
    def arrays(ep):
        # Give held-out observations a distinctive magnitude to detect leakage.
        offset=100. if ep in val else float(ep)
        pose=np.tile([0,0,.2,0,0,0,1.],(65,1));pose[:,0]=np.arange(65)*.0001
        source=np.array(['expert']*65) if ep>=4 else np.array(['model']*10+['expert']*35+['hold']*20)
        segment=np.zeros(65,dtype=np.int64) if ep>=4 else np.array([0]*10+[1]*35+[2]*20)
        return {STATE:np.full((65,6),offset),WRENCH:np.full((65,6),offset),
            'action':pose,'tcp_pose':pose.copy(),'action_source':source,'action_segment':segment,
            'human_active':source=='expert','simulation_time':np.arange(65)*.05}
    def sampled(path):
        ep=int(Path(path).stem);a=arrays(ep)
        mask,train,validation,report=rollout_windows(a['action_source'],a['action_segment'],a['human_active'],
            a['simulation_time'],a['action'],a[WRENCH],a['tcp_pose'])
        return a['tcp_pose'],mask,train,validation,report
    monkeypatch.setattr('rollout_sampling.sample_rollout_file',sampled)
    def prepared(source,output,entry,threads,**kwargs):
        ep=entry['episode_index'];directory=root/'cache';directory.mkdir(exist_ok=True)
        numeric=directory/f'{ep}.npz';images=directory/f'{ep}.images'
        if not numeric.exists():np.savez(numeric,**arrays(ep))
        images.write_bytes(b'no image decode needed for index audit')
        import os
        return {'episode_index':ep,'source_attempt':entry['attempt'],'length':65,
            'source_sha256':entry['validation']['parquet_sha256'],
            'numeric':{'path':os.path.relpath(numeric,output),'bytes':numeric.stat().st_size,'sha256':file_digest(numeric)},
            'images':{'path':os.path.relpath(images,output),'bytes':images.stat().st_size,'sha256':file_digest(images)}}
    monkeypatch.setattr('prepare_expert_only.prepare_episode',prepared)
    entries=[]
    for ep in range(6):
        a=arrays(ep)
        entries.append({'accepted':True,'episode_index':ep,'attempt':ep,'parquet':f'{ep}.parquet',
            'validation':{'frames':65,'parquet_sha256':str(ep).zfill(64)},
            'official_score':{'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}},
            'recorder':{'policy_return':True,'writer_error':None},
            'expert_anchor_frames':expert_anchor_frames(a['action_source'],a['action_segment'],a['human_active'],a['simulation_time'],source='expert')})
    (root/'_collection/manifest.json').write_text(json.dumps({'format':EXPERT_FORMAT,'task':TASK,'attempts':entries}))
    return root,val,arrays


def test_new_snapshot_uses_direct_and_assisted_episodes_and_no_val_statistics(tmp_path,monkeypatch):
    root,val,arrays=collection(tmp_path,monkeypatch)
    with patch('torch.load',side_effect=AssertionError('No checkpoint allowed')):
        index=prepare_expert_only(root,eval_samples=10,seed=43,retain_model_motion=True)
    assert len(index['episodes'])==6 and index['preparation_settings']['seed']==43
    assert index['normalization_source']=='current training episodes only'
    assert 'normalization_parent_index_sha256' not in index
    assert index['normalization_training_episode_ids']==sorted(set(range(6))-set(val))
    train=index['normalization_training_episode_ids']
    assert index['stats'][STATE]['mean']==pytest.approx([np.mean(train)]*6)
    assert index['stats'][STATE]['count']==[65*5]
    assert len(index['validation_anchors'])==10
    for e in index['episodes']:
        expected=45 if e['episode_index']>=4 else 16
        assert len(e['expert_validation_frames'])==expected
    # Cache reuse returns the exact same fixed membership and VAL anchors.
    again=prepare_expert_only(root,eval_samples=10,seed=43,retain_model_motion=True)
    assert again==index
    # The loader rejects an index claiming normalization came from held-out episodes.
    value=copy.deepcopy(index);directory=Path(value.pop('_directory'))
    value['normalization_training_episode_ids'].append(val[0])
    value['index_content_sha256']=index_digest(value)
    path=directory/'bad.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='exactly the training'):read_index(path)


def test_auto_preparation_binds_saved_path_and_resume_never_repartitions(tmp_path,monkeypatch):
    root,_,_=collection(tmp_path,monkeypatch)
    cfg=AlternatingResidualFlowConfig(dataset_root=str(root),device='cpu',amp='none',
        from_scratch=True,init_from=None,pretrained_backbone=False,expert_only=True,
        index_path='auto',eval_samples=10,seed=43,base_act_config={'index_path':'auto'})
    index=trainer.prepare_training_index(cfg)
    assert cfg.index_path.endswith('/index.json') and cfg.base_act_config['index_path']==cfg.index_path
    reference={'data_index_sha256':index['index_content_sha256'],'stats':index['stats']}
    with patch('prepare_expert_only.prepare_expert_only',side_effect=AssertionError('Resume must keep its split')):
        assert trainer.prepare_training_index(cfg,resume=reference)==index
    cfg.index_path='auto'
    with pytest.raises(ValueError,match='without previous weights'):
        trainer.prepare_training_index(cfg,resume=reference)


def test_new_config_keeps_both_models_trainable_and_separate_output():
    path=Path(__file__).resolve().parents[1]/'configs/sfp0_act_flow_fresh_v2.json'
    with patch.object(sys,'argv',['train','--check-data']):
        _,cfg,resume=trainer.parse_args(path)
    assert cfg.index_path=='auto' and cfg.seed==43 and resume is None
    assert cfg.from_scratch and cfg.init_from is None and not cfg.pretrained_backbone
    assert cfg.eval_samples==1500 and cfg.eval_every==cfg.save_every==500
    assert cfg.output_dir.endswith('aic_sfp0_act_flow_fresh_v2')
