"""Preserve an alternating pair exactly while training only its residual Flow."""
from dataclasses import asdict,fields
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from residual_flow_multiview import (ResidualFlowConfig,SupervisedResidualFlow,
    bind_base_act,initialize_residual_from_act,ALTERNATING_FORMAT)
from train_sfp0_act_multiview import load_initialization,parse_args,set_optimizer_learning_rates
from test_alternating_residual_flow import setup as alternating_setup,states,assert_states
from test_residual_flow import residual_batch
from test_act_multiview import stats


@pytest.fixture(scope='module',autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def setup():
    source,source_cfg,_=alternating_setup()
    source.act_update_count.fill_(10000);source.flow_update_count.fill_(10000)
    payload={'format':ALTERNATING_FORMAT,'model':source.state_dict(),'config':asdict(source_cfg),
        'stats':stats(),'data_index_sha256':'data','step':10000,'smoke_test':False,
        'camera_order':list(source.cameras),'action_offset':1,'_initialization_sha256':'b'*64}
    values={f.name:getattr(source_cfg,f.name) for f in fields(ResidualFlowConfig)}
    values.update(policy_type='residual_flow',steps=5000,learning_rate=3e-6,backbone_lr=3e-7,
        warmup_steps=0,minimum_lr_ratio=1.,retain_model_motion=False,expert_only=True)
    cfg=ResidualFlowConfig(**values)
    bind_base_act(cfg,payload)
    model=SupervisedResidualFlow(cfg,stats(),initialize_backbone=False)
    report=initialize_residual_from_act(model,payload)
    return source.eval(),model,cfg,payload,report


def test_pair_initialization_preserves_full_flow_and_current_act_predictions():
    source,model,cfg,payload,report=setup()
    assert report['flow_weights_preserved'] and report['optimizer_restarted']
    assert cfg.base_act_step==10000 and cfg.base_act_sha256=='b'*64
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,payload['model'][key],rtol=0,atol=0)
    b=residual_batch();noise=torch.randn(2,20,7)
    model.eval()
    torch.testing.assert_close(model.predict_action_chunk(b,noise=noise),
                              source.predict_action_chunk(b,noise=noise),rtol=1e-5,atol=1e-7)


def test_only_flow_updates_and_frozen_act_weights_buffers_remain_identical():
    _,model,cfg,_,_=setup()
    model.train();before=states(model.base_act)
    head_before=model.policy.model.action_head.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.policy.get_optim_params(),lr=cfg.learning_rate)
    parameters={id(p) for g in optimizer.param_groups for p in g['params']}
    assert not any(id(p) in parameters for p in model.base_act.parameters())
    assert all(not p.requires_grad for p in model.base_act.parameters())
    loss,_=model(residual_batch());loss.backward();optimizer.step()
    assert not model.base_act.training and all(p.grad is None for p in model.base_act.parameters())
    assert_states(model.base_act,before)
    assert not torch.equal(head_before,model.policy.model.action_head.weight)


def test_full_pair_source_only_accepted_by_frozen_flow_entrypoint(tmp_path):
    _,_,_,payload,_=setup()
    path=tmp_path/'pair.pt';torch.save(payload,path)
    loaded,provenance=load_initialization(path,expected_policy='residual_flow')
    assert loaded['step']==10000 and len(provenance['sha256'])==64
    with pytest.raises(ValueError,match='ACT checkpoint'):
        load_initialization(path,expected_policy='act')


def test_warm_start_rejects_attention_semantics_changes_even_if_weights_fit():
    _,_,cfg,payload,_=setup()
    cfg.n_heads=2
    with pytest.raises(ValueError,match='n_heads'):
        bind_base_act(cfg,payload)


def test_default_uses_same_val_and_low_constant_lr_without_warmup():
    root=Path(__file__).resolve().parents[1]
    path=root/'configs/sfp0_flow_only_after_scratch.json'
    config=json.loads(path.read_text())
    assert config['eval_samples']==1500 and config['eval_every']==500
    assert config['init_from'].endswith('aic_sfp0_alternating_residual_flow_scratch_v1/checkpoints/best.pt')
    assert config['output_dir'].endswith('aic_sfp0_flow_only_after_scratch_v1')
    with patch.object(sys,'argv',['train','--config',str(path),'--check-data']):
        _,cfg,_=parse_args(expected_policy='residual_flow')
    p=torch.nn.Parameter(torch.ones(1));q=torch.nn.Parameter(torch.ones(1))
    optimizer=torch.optim.AdamW([{'params':[p]},{'params':[q]}])
    for step in [1,5000]:
        set_optimizer_learning_rates(optimizer,cfg,step)
        assert optimizer.param_groups[0]['lr']==3e-6
        assert optimizer.param_groups[1]['lr']==3e-7
