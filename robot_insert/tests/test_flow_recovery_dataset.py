"""A direct-demo ACT can stay frozen while Flow learns a different recovery set."""
from dataclasses import asdict,replace
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_sfp0_direct_act import select_recovery
from residual_flow_multiview import ResidualFlowConfig,load_residual_index
from test_fresh_act_flow import collection
from test_act_multiview import stats
from train_sfp0_act_multiview import parse_args,set_optimizer_learning_rates


def test_selection_keeps_only_expert_recovery_success_and_does_not_promote_model_prefix():
    from human_dataset import EXPERT_FORMAT
    score={'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}}
    recovery={'accepted':True,'episode_index':1,'official_score':score,'expert_validation_frames':[9,10],
        'recorder':{'controller':'model_and_official_expert','policy_return':True,'writer_error':None}}
    direct=copy.deepcopy(recovery);direct['episode_index']=2;direct['recorder']['controller']='official_cheatcode_only'
    model=copy.deepcopy(recovery);model['episode_index']=3;model['expert_validation_frames']=[]
    another=copy.deepcopy(recovery);another['episode_index']=4
    manifest={'format':EXPERT_FORMAT,'attempts':[recovery,direct,model,another]}
    selected=select_recovery(manifest)
    assert [e['episode_index'] for e in selected]==[1,4]
    assert all(e['recorder']['controller']=='model_and_official_expert' for e in selected)


def test_new_flow_data_keeps_act_statistics_and_resume_cannot_silently_refresh(tmp_path,monkeypatch):
    root,_,_=collection(tmp_path,monkeypatch)
    initial={'stats':stats(),'data_index_sha256':'a'*64}
    cfg=ResidualFlowConfig(dataset_root=str(root),index_path='auto',new_dataset=True,
        init_from='new_act.pt',device='cpu',amp='none',pretrained_backbone=False,
        expert_only=True,eval_samples=10,seed=43)
    index=load_residual_index(cfg,initial=initial)
    assert index['stats']==initial['stats'] and index['normalization_parent_index_sha256']=='a'*64
    assert index['index_content_sha256']!=initial['data_index_sha256']
    assert cfg.index_path.endswith('/index.json')
    reference={'stats':index['stats'],'data_index_sha256':index['index_content_sha256']}
    with patch('prepare_expert_only.prepare_expert_only',side_effect=AssertionError('Keep frozen data on resume')):
        assert load_residual_index(cfg,resume=reference)==index
    changed=copy.deepcopy(reference);changed['data_index_sha256']='b'*64
    with pytest.raises(ValueError,match='same snapshot'):load_residual_index(cfg,resume=changed)
    cfg.new_dataset=False
    with pytest.raises(ValueError,match='same snapshot'):load_residual_index(cfg,initial=initial)


def test_entrypoint_freezes_new_direct_act_and_uses_only_curated_recovery_data():
    path=Path(__file__).resolve().parents[1]/'configs/sfp0_flow_recovery_after_direct_act.json'
    with patch.object(sys,'argv',['train','--config',str(path),'--check-data']):
        _,cfg,_=parse_args(expected_policy='residual_flow')
    assert cfg.new_dataset and cfg.index_path=='auto'
    assert cfg.init_from.endswith('aic_sfp0_act_direct_scratch_v1/checkpoints/best.pt')
    assert cfg.dataset_root.endswith('aic_sfp0_recovery_only_v1')
    assert cfg.expert_only and not cfg.retain_model_motion and cfg.eval_samples==1500


def test_extend_recovery_flow_preserves_schedule_optimizer_and_frozen_contract(tmp_path):
    from residual_flow_multiview import FORMAT
    cfg=ResidualFlowConfig(device='cpu',amp='none',steps=10000,warmup_steps=200,
        learning_rate=3e-5,backbone_lr=3e-6,expert_only=True,eval_samples=1500,
        index_path='/mnt/d/frozen/index.json',base_act_step=3000,
        base_act_sha256='a'*64,base_act_config={'frozen':'architecture'})
    parameters=[torch.nn.Parameter(torch.tensor([1.])),torch.nn.Parameter(torch.tensor([2.]))]
    optimizer=torch.optim.AdamW([{'params':[parameters[0]],'lr':3e-5},
                                 {'params':[parameters[1]],'lr':3e-6}])
    for p in parameters:p.grad=torch.ones_like(p)
    optimizer.step()
    payload={'format':FORMAT,'optimizer':optimizer.state_dict(),'config':asdict(cfg),'step':10000}
    # Old checkpoints have no schedule-horizon field.
    payload['config'].pop('lr_schedule_steps')
    path=tmp_path/'last.pt';torch.save(payload,path)
    with patch.object(sys,'argv',['train','--resume',str(path),'--steps','20000']):
        _,extended,resume=parse_args(expected_policy='residual_flow')
    assert extended.steps==20000 and extended.lr_schedule_steps==10000
    assert extended.base_act_config==cfg.base_act_config and extended.index_path==cfg.index_path
    assert extended.base_act_sha256==cfg.base_act_sha256 and extended.base_act_step==3000
    optimizer.load_state_dict(resume['optimizer'])
    prior=copy.deepcopy(optimizer.state_dict()['state'])
    for step in (10001,15000,20000):
        set_optimizer_learning_rates(optimizer,extended,step)
        assert optimizer.param_groups[0]['lr']==pytest.approx(3e-6)
        assert optimizer.param_groups[1]['lr']==pytest.approx(3e-7)
    for i,values in optimizer.state_dict()['state'].items():
        for k,v in values.items():torch.testing.assert_close(v,prior[i][k],rtol=0,atol=0)
    # Checkpoint roundtrip and a second extension keep the original horizon.
    payload['config']=asdict(extended);payload['step']=15000;torch.save(payload,path)
    for options,expected in (([],20000),(['--steps','25000'],25000)):
        with patch.object(sys,'argv',['train','--resume',str(path),*options]):
            _,continued,_=parse_args(expected_policy='residual_flow')
        assert continued.steps==expected and continued.lr_schedule_steps==10000
    for options in (['--steps','18000'],['--steps','15000'],['--position-scale-m','.02'],
                    ['--index-path','/mnt/d/other/index.json']):
        with patch.object(sys,'argv',['train','--resume',str(path),*options]):
            with pytest.raises(SystemExit):parse_args(expected_policy='residual_flow')
