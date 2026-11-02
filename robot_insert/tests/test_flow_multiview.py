"""Four-view masked CFM, observation-only transfer, reproducible generation."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from act_multiview import MultiViewACT,MultiViewConfig,FOUR_CAMERAS
from act_objectives import evaluate_pose_chunks
from flow_matching import ot_path,anchor_noise,integrate_velocity
from flow_multiview import (MultiViewFlowConfig,MultiViewFlow,masked_velocity_mse,
    initialize_flow_encoder_from_act,FORMAT)
from ftact import load_checkpoint
from train_sfp0_act_multiview import checkpoint,parse_args
from test_act_multiview import stats,batch


@pytest.fixture(scope='module',autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def config():
    return MultiViewFlowConfig(device='cpu',amp='none',dim_model=32,n_heads=4,
        dim_feedforward=64,n_encoder_layers=1,n_decoder_layers=1,
        pretrained_backbone=False,dropout=0.,steps=10,ode_steps=2)


def flow_batch():
    b=batch();b['action_mask']=torch.ones(2,20,dtype=torch.bool)
    b['episode_index']=torch.tensor([0,1]);b['frame_index']=torch.tensor([5,10])
    return b


def test_paper_ot_path_derivative_and_exact_constant_velocity_integration():
    noise=torch.randn(2,20,7);action=torch.randn_like(noise)
    time=torch.tensor([.3,.7]);sigma=1e-4
    x,u=ot_path(action,noise,time,sigma)
    torch.testing.assert_close(x,(1-(1-sigma)*time[:,None,None])*noise+time[:,None,None]*action)
    torch.testing.assert_close(u,action-(1-sigma)*noise)
    for solver in ['euler','heun']:
        sampled=integrate_velocity(lambda x,t:u,noise,10,solver)
        torch.testing.assert_close(sampled,action+sigma*noise,atol=2e-6,rtol=2e-5)


def test_masked_targets_have_no_loss_gradient_or_attention_input_influence():
    model=MultiViewFlow(config(),stats(),initialize_backbone=False).eval()
    b=flow_batch();b['action_mask'][:,10:]=False
    state=torch.get_rng_state()
    first,_=model(b)
    modified=dict(b);modified['action']=b['action'].clone();modified['action'][:,10:]=float('nan')
    torch.set_rng_state(state)
    second,_=model(modified)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    pred=torch.ones(1,20,7,requires_grad=True);target=torch.zeros_like(pred)
    mask=torch.zeros(1,20,dtype=torch.bool);mask[:,:5]=True
    target[:,5:]=float('nan')
    loss=masked_velocity_mse(pred,target,mask);loss.backward()
    assert loss.item()==1 and pred.grad[:,:5].abs().sum()>0 and not pred.grad[:,5:].any()
    with pytest.raises(ValueError):masked_velocity_mse(pred,target,~torch.ones_like(mask))


def test_transfer_only_observation_encoder_preserves_fresh_flow_generator():
    cfg=config()
    act_cfg=MultiViewConfig(**{k:v for k,v in asdict(cfg).items() if k not in
        ['sigma_min','ode_steps','ode_solver','validation_noise_seed']})
    act_cfg.policy_type='act'
    act=MultiViewACT(act_cfg,stats(),initialize_backbone=False)
    flow=MultiViewFlow(cfg,stats(),initialize_backbone=False)
    initial={k:v.clone() for k,v in flow.state_dict().items()}
    report=initialize_flow_encoder_from_act(flow,act.state_dict())
    assert report['transferred_tensors']>100
    for key,value in flow.state_dict().items():
        if key.startswith(('policy.model.backbone.','policy.model.encoder.','policy.model.encoder_')):
            torch.testing.assert_close(value,act.state_dict()[key],rtol=0,atol=0)
        else:torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
    assert not torch.equal(flow.policy.model.action_head.weight,act.policy.model.action_head.weight)


def test_common_xyz_action_scale_does_not_downweight_z_or_change_wrench_stats():
    values=stats();values['action']['std'][2]=.08
    model=MultiViewFlow(config(),values,initialize_backbone=False)
    torch.testing.assert_close(model.normalizer.action_std[:3],torch.full((3,),.01))
    origin=model.normalizer.action_mean[None,None].expand(1,20,7).clone()
    errors=[]
    for axis in range(3):
        changed=origin.clone();changed[...,axis]+=.001
        errors.append((model.normalizer.actions(changed)-model.normalizer.actions(origin))[...,axis])
    torch.testing.assert_close(errors[0],errors[1]);torch.testing.assert_close(errors[1],errors[2])
    expected=torch.tensor(values['observation.environment_state']['std'],dtype=torch.float32)
    torch.testing.assert_close(model.normalizer.state_std[6:],expected)


def test_fourth_view_force_and_time_receive_gradients():
    model=MultiViewFlow(config(),stats(),initialize_backbone=False).train()
    b=flow_batch();b[FOUR_CAMERAS[-1]].requires_grad_(True)
    loss,metrics=model(b);loss.backward()
    assert torch.isfinite(loss) and metrics['flow_valid_steps']==40
    assert b[FOUR_CAMERAS[-1]].grad.norm()>0
    core=model.policy.model
    assert core.encoder_robot_state_input_proj.weight.grad[:,6:].norm()>0
    assert core.flow_action_input_proj.weight.grad.norm()>0
    assert core.flow_time_mlp[-1].weight.grad.norm()>0


def test_ode_caches_four_view_encoder_once_and_never_reads_future_actions():
    model=MultiViewFlow(config(),stats(),initialize_backbone=False).eval();b=flow_batch()
    noise=anchor_noise([0,1],[5,10],10042)
    with patch.object(model,'encode_observations',wraps=model.encode_observations) as encode:
        first=model.predict_action_chunk(b,noise=noise,steps=5)
    assert encode.call_count==1 and first.shape==(2,20,7)
    torch.testing.assert_close(first[...,3:].norm(dim=-1),torch.ones(2,20))
    b.pop('action');b.pop('action_mask');b.pop('action_is_pad')
    second=model.predict_action_chunk(b,noise=noise,steps=5)
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_validation_is_actual_generation_with_fixed_anchor_noise_and_keeps_rng():
    cfg=config();cfg.eval_samples=2
    model=MultiViewFlow(cfg,stats(),initialize_backbone=False).eval();b=flow_batch()
    rng=torch.get_rng_state().clone()
    first=evaluate_pose_chunks(model,[b],cfg)
    assert torch.equal(rng,torch.get_rng_state())
    reverse={k:v.flip(0) for k,v in b.items()}
    second=evaluate_pose_chunks(model,[reverse],cfg)
    assert first['position_l2_m']==pytest.approx(second['position_l2_m'],rel=1e-6)
    assert first['generation']=='conditional_flow_matching' and first['ode_steps']==2
    assert first['validation_samples_per_anchor']==1 and first['valid_action_steps']==40


def test_resume_and_inference_checkpoint_roundtrip_preserve_flow_contract(tmp_path):
    cfg=config();model=MultiViewFlow(cfg,stats(),initialize_backbone=False).eval();b=flow_batch()
    noise=anchor_noise([0,1],[5,10],10042)
    expected=model.predict_action_chunk(b,noise=noise)
    optimizer=torch.optim.AdamW(model.policy.get_optim_params(),lr=cfg.learning_rate)
    cursor=NS(state_dict=lambda:{'epoch':0,'samples_consumed':2,'batches_consumed':1})
    index={'stats':stats(),'task':{'target_module':'nic_card_mount_0'},'index_content_sha256':'test'}
    path=tmp_path/'last.pt'
    checkpoint(path,model,optimizer,torch.amp.GradScaler('cuda',enabled=False),cfg,index,
        1,cursor,1.,.01,{'loss':1.},1,True,True)
    loaded,payload=load_checkpoint(path)
    assert payload['format']==FORMAT and payload['reference_paper'].endswith('2210.02747')
    assert payload['execution']['temporal_ensemble'] is False
    torch.testing.assert_close(loaded.predict_action_chunk(b,noise=noise),expected,rtol=0,atol=0)
    with patch.object(sys,'argv',['train','--resume',str(path)]):
        _,resumed,_=parse_args(expected_policy='flow')
    assert resumed.ode_steps==2
    with patch.object(sys,'argv',['train','--resume',str(path),'--ode-steps','5']):
        with pytest.raises(SystemExit):parse_args(expected_policy='flow')
    with patch.object(sys,'argv',['train','--resume',str(path)]):
        with pytest.raises(SystemExit):parse_args(expected_policy='act')


def test_config_reuses_same_source_snapshot_and_validation_budget():
    root=Path(__file__).resolve().parents[1]
    act=json.loads((root/'configs/sfp0_act_expert_finetune.json').read_text())
    flow=json.loads((root/'configs/sfp0_flow_expert_finetune.json').read_text())
    for key,value in act.items():
        if key not in ['policy_type','output_dir']:assert flow[key]==value
    with patch.object(sys,'argv',['train','--config',str(root/'configs/sfp0_flow_expert_finetune.json'),'--check-data']):
        _,cfg,_=parse_args(expected_policy='flow')
    assert cfg.sigma_min==1e-4 and cfg.eval_samples==1500
