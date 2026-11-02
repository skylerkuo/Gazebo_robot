"""Frozen same-observation ACT, expert-only targets, residual composition/reload."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from act_multiview import MultiViewACT,MultiViewConfig,MultiViewDataset,FOUR_CAMERAS,ROLLOUT_INDEX_FORMAT
from residual_flow_multiview import (ResidualFlowConfig,ResidualExpertDataset,
    SupervisedResidualFlow,bind_base_act,initialize_residual_from_act,load_residual_index,FORMAT)
from ftact import load_checkpoint,unit_tcp_quaternions
from train_sfp0_act_multiview import checkpoint,parse_args
from test_act_multiview import stats,batch


@pytest.fixture(scope='module',autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def setup():
    common=dict(device='cpu',amp='none',dim_model=32,n_heads=4,dim_feedforward=64,
        n_encoder_layers=1,n_decoder_layers=1,pretrained_backbone=False,dropout=0.,steps=10)
    base_cfg=MultiViewConfig(**common)
    base=MultiViewACT(base_cfg,stats(),initialize_backbone=False)
    cfg=ResidualFlowConfig(**common,expert_only=True,retain_model_motion=False,ode_steps=2,
        residual_scale=.5,index_path='/mnt/d/frozen/index.json')
    payload={'format':'aic-sfp0-act-multiview-v1','config':asdict(base_cfg),
        'model':base.state_dict(),'step':5,'_initialization_sha256':'a'*64,
        'data_index_sha256':'data','stats':stats()}
    bind_base_act(cfg,payload)
    model=SupervisedResidualFlow(cfg,stats(),initialize_backbone=False)
    initialize_residual_from_act(model,payload)
    return model,cfg,payload


def residual_batch():
    b=batch();b['action']=unit_tcp_quaternions(b['action'])
    b['action_mask']=torch.ones(2,20,dtype=torch.bool)
    b['episode_index']=torch.tensor([0,1]);b['frame_index']=torch.tensor([9,10])
    return b


def test_expert_window_selection_keeps_handoff_observation_but_no_learner_targets(monkeypatch):
    index={'_directory':'unused','episodes':[{'episode_index':0,'split':'train','length':50,
        'train_anchor_frames':list(range(30)),'expert_validation_frames':[9,10]}]}
    numeric={'action_source':np.array(['model']*10+['expert']*30+['hold']*10),
        'action_segment':np.array([0]*10+[1]*30+[2]*10),'simulation_time':np.arange(50)*.05}
    def sample(self,i):
        ep,f=self.anchors[i];self.numeric[ep]=numeric
        return {'action':torch.zeros(20,7),'action_mask':torch.ones(20,dtype=torch.bool)}
    monkeypatch.setattr(MultiViewDataset,'__getitem__',sample)
    ds=ResidualExpertDataset(index,'train')
    assert ds.anchors==[[0,9],[0,10]]
    ds[0]  # Current observation may be the learner's final tick; all future actions are expert.
    numeric['action_source'][15]='model'
    with pytest.raises(ValueError):ds[0]
    numeric['action_source'][15]='expert';numeric['simulation_time'][14:]+=.05
    with pytest.raises(ValueError):ds[0]


def test_fixed_act_weights_buffers_and_gradients_never_update():
    model,cfg,payload=setup();b=residual_batch();model.train()
    assert not model.base_act.training and all(not p.requires_grad for p in model.base_act.parameters())
    before={k:v.clone() for k,v in model.base_act.state_dict().items()}
    optimizer=torch.optim.AdamW(model.policy.get_optim_params(),lr=1e-4)
    groups={id(p) for group in optimizer.param_groups for p in group['params']}
    assert not any(id(p) in groups for p in model.base_act.parameters())
    loss,_=model(b);loss.backward()
    assert all(p.grad is None for p in model.base_act.parameters())
    assert model.policy.model.flow_base_action_input_proj.weight.grad.norm()>0
    optimizer.step()
    for key,value in model.base_act.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_transferred_flow_encoder_copies_values_but_remains_independently_trainable():
    model,cfg,payload=setup()
    assert cfg.flow_encoder_init_from_act
    prefixes=('policy.model.backbone.','policy.model.encoder.','policy.model.encoder_')
    copied={k:v for k,v in model.state_dict().items() if k.startswith(prefixes)}
    assert copied and any(k.startswith('policy.model.encoder_') for k in copied)
    base=model.base_act.state_dict()
    for name,value in copied.items():
        torch.testing.assert_close(value,payload['model'][name],rtol=0,atol=0)
        assert value.data_ptr()!=base[name].data_ptr()
    params=dict(model.named_parameters())
    assert all(params[k].requires_grad for k in copied if k in params)
    model.train()
    loss,_=model(residual_batch());loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0
               for k,p in params.items() if k.startswith('policy.model.encoder_'))
    assert all(p.grad is None for p in model.base_act.parameters())


def test_completely_random_flow_does_not_import_the_new_act_encoder_or_old_flow():
    _,cfg,payload=setup()
    cfg.flow_encoder_init_from_act=False
    model=SupervisedResidualFlow(cfg,stats(),initialize_backbone=False)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    report=initialize_residual_from_act(model,payload)
    assert report['flow_initialization']=='fully_random' and not report['encoder_copied_from_act']
    for key,value in model.policy.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    for key,value in model.base_act.state_dict().items():
        torch.testing.assert_close(value,payload['model'][key],rtol=0,atol=0)
    assert not model.base_act.training and not any(p.requires_grad for p in model.base_act.parameters())
    base_backbone=model.base_act.policy.model.backbone.state_dict()
    flow_backbone=model.policy.model.backbone.state_dict()
    key=next(k for k in base_backbone if k.endswith('weight'))
    assert not torch.equal(flow_backbone[key],base_backbone[key])
    loss,_=model(residual_batch());loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None for p in model.base_act.parameters())
    bad=dict(payload,format='aic-sfp0-alternating-act-residual-flow-multiview-v1')
    with pytest.raises(ValueError,match='ACT-only'):initialize_residual_from_act(model,bad)


def test_same_observation_residual_reconstructs_expert_and_aligns_quaternion_sign():
    model,cfg,_=setup();model.eval();b=residual_batch()
    base=model.base_normalized_action(b)
    target=model.residual_target(b['action'],base)
    reconstructed=model.normalizer.unnormalize_actions(base+cfg.residual_scale*target)
    torch.testing.assert_close(reconstructed[...,:3],b['action'][...,:3])
    dot=(reconstructed[...,3:]*b['action'][...,3:]).sum(-1).abs()
    torch.testing.assert_close(dot,torch.ones_like(dot))
    physical=model.normalizer.unnormalize_actions(base)
    opposite=physical.clone();opposite[...,3:]*=-1
    torch.testing.assert_close(model.residual_target(opposite,base),torch.zeros_like(base),atol=2e-5,rtol=0)


def test_oracle_residual_ode_composes_with_act_to_reproduce_expert():
    model,cfg,_=setup();model.eval();b=residual_batch()
    base=model.base_normalized_action(b)
    target=model.residual_target(b['action'],base)
    noise=torch.randn_like(target)
    with patch.object(model,'velocity',side_effect=lambda x,t,c:target-noise):
        actual=model.predict_action_chunk(b,noise=noise,steps=5)
    torch.testing.assert_close(actual[...,:3],b['action'][...,:3],atol=1e-6,rtol=1e-5)
    dot=(actual[...,3:]*b['action'][...,3:]).sum(-1).abs()
    torch.testing.assert_close(dot,torch.ones_like(dot),atol=1e-5,rtol=1e-5)


def test_zero_residual_and_future_label_independence():
    model,cfg,_=setup();model.eval();b=residual_batch()
    noise=torch.randn(2,20,7)
    first=model.predict_action_chunk(b,noise=noise)
    changed={k:v for k,v in b.items() if k not in ['action','action_mask','action_is_pad']}
    torch.testing.assert_close(model.predict_action_chunk(changed,noise=noise),first,rtol=0,atol=0)
    model.residual_enabled=False
    torch.testing.assert_close(model.predict_action_chunk(changed),model.base_act.predict_action_chunk(changed),atol=1e-7,rtol=1e-6)
    model.residual_enabled=True
    with patch('residual_flow_multiview.integrate_velocity',return_value=torch.zeros_like(noise)):
        torch.testing.assert_close(model.predict_action_chunk(changed,noise=noise),model.base_act.predict_action_chunk(changed),atol=1e-7,rtol=1e-6)


def test_checkpoint_embeds_base_and_resumes_without_source_file(tmp_path):
    model,cfg,_=setup();model.eval();b=residual_batch();noise=torch.randn(2,20,7)
    expected=model.predict_action_chunk(b,noise=noise)
    optimizer=torch.optim.AdamW(model.policy.get_optim_params())
    index={'stats':stats(),'task':{'target_module':'nic_card_mount_0'},'index_content_sha256':'data'}
    cursor=NS(state_dict=lambda:{'epoch':0,'samples_consumed':2,'batches_consumed':1})
    path=tmp_path/'last.pt'
    checkpoint(path,model,optimizer,torch.amp.GradScaler('cuda',enabled=False),cfg,index,
        1,cursor,1.,.01,{'loss':1.},1,True,True)
    loaded,payload=load_checkpoint(path)
    assert payload['format']==FORMAT and payload['base_act_contract']['embedded_in_model']
    assert payload['base_act_contract']['sha256']=='a'*64
    assert payload['execution']['temporal_ensemble'] is False
    assert loaded.base_act.training is False and not any(p.requires_grad for p in loaded.base_act.parameters())
    torch.testing.assert_close(loaded.predict_action_chunk(b,noise=noise),expected,rtol=0,atol=0)
    with patch.object(sys,'argv',['train','--resume',str(path)]):
        _,resumed,_=parse_args(expected_policy='residual_flow')
    assert resumed.base_act_step==5 and resumed.residual_scale==.5
    with patch.object(sys,'argv',['train','--resume',str(path),'--residual-scale','1']):
        with pytest.raises(SystemExit):parse_args(expected_policy='residual_flow')
    with patch.object(sys,'argv',['train','--resume',str(path)]):
        with pytest.raises(SystemExit):parse_args(expected_policy='flow')


def test_frozen_snapshot_must_match_base_act_and_training_selection(monkeypatch):
    _,cfg,payload=setup()
    index={'format':ROLLOUT_INDEX_FORMAT,'index_content_sha256':'data','stats':stats(),
        'dataset_root':cfg.dataset_root,'validation_samples':cfg.eval_samples}
    monkeypatch.setattr('residual_flow_multiview.read_index',lambda *a,**k:index)
    assert load_residual_index(cfg,payload) is index
    wrong=copy.deepcopy(payload);wrong['data_index_sha256']='other'
    with pytest.raises(ValueError,match='same snapshot'):load_residual_index(cfg,wrong)
    cfg.retain_model_motion=True
    with pytest.raises(ValueError,match='expert-only'):cfg.validate()


def test_incomplete_targets_cannot_train_residual():
    model,_,_=setup();b=residual_batch();b['action_mask'][:,5]=False
    with pytest.raises(ValueError,match='complete valid expert'):model(b)
