"""Four-view PACE uses ACT's same split, masks, initialization and physical loss."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
from unittest.mock import Mock,patch
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from act_multiview import MultiViewConfig,MultiViewACT,MultiViewDataset,read_index,index_digest
from pace_multiview import (MultiViewPACEConfig,MultiViewPACE,MultiViewPACEDataset,
    initialize_pace_from_act,FORMAT)
from pace_act import PHASE,HISTORY
from ftact import WRENCH,load_checkpoint
from train_sfp0_act_multiview import checkpoint,parse_args
from train_sfp0_pace import gradient_metrics
from test_act_multiview import stats,batch


def config():
    return MultiViewPACEConfig(device='cpu',amp='none',dim_model=32,n_heads=4,
        dim_feedforward=64,n_encoder_layers=1,n_decoder_layers=1,pretrained_backbone=False,
        dropout=0.,batch_size=2)


def pace_batch():
    b=batch();history=torch.randn(2,4,6)
    b[WRENCH]=history[:,-1].clone();b[PHASE]=history.mean(1);b[HISTORY]=history
    b['action_mask']=torch.ones(2,20,dtype=torch.bool);b['action_mask'][:,10:]=False
    return b


@pytest.fixture(scope='module',autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_configs_share_data_validation_budget_and_initialization():
    act=json.loads((ROOT/'configs/sfp0_act_expert_finetune.json').read_text())
    pace=json.loads((ROOT/'configs/sfp0_pace_expert_finetune.json').read_text())
    for key,value in act.items():
        if key not in ('policy_type','output_dir'):assert pace[key]==value
    assert act['comparison_index_path'] and act['eval_samples']==1500 and act['eval_every']==500
    with patch.object(sys,'argv',['train','--config',str(ROOT/'configs/sfp0_pace_expert_finetune.json'),'--check-data']):
        # Hardware-free preflight must work even while the GPU is occupied/unavailable.
        _,cfg,_=parse_args(expected_policy='pace')
    assert cfg.policy_type=='pace' and cfg.tactile_history==4 and cfg.correction_horizon==5


def test_causal_context_survives_handoff_resets_episodes_and_fills_only_known_gaps(monkeypatch):
    data={e:{WRENCH:np.tile(np.arange(40,dtype=np.float32)[:,None]+e*100,(1,6)),
        'simulation_time':np.arange(40)*.05} for e in (0,1)}
    index={'_directory':'unused','episodes':[{'episode_index':e,'split':'train','length':40,
        'train_anchor_frames':[0,2,3,5]} for e in (0,1)],'validation_anchors':[]}
    def base(self,i):
        ep,frame=self.anchors[i];self.numeric[ep]=data[ep]
        return {WRENCH:torch.from_numpy(data[ep][WRENCH][frame].copy()),'action_mask':torch.ones(20,dtype=torch.bool)}
    monkeypatch.setattr(MultiViewDataset,'__getitem__',base)
    dataset=MultiViewPACEDataset(index,'train',alpha=.5)
    at3=dataset[2]
    np.testing.assert_allclose(at3[PHASE],2.125)
    np.testing.assert_array_equal(at3[HISTORY][:,0],[0,1,2,3])
    np.testing.assert_array_equal(dataset[4][HISTORY],np.full((4,6),100))
    data[0][WRENCH][4:]+=1000
    changed=MultiViewPACEDataset(index,'train',alpha=.5)[2]
    torch.testing.assert_close(at3[PHASE],changed[PHASE]);torch.testing.assert_close(at3[HISTORY],changed[HISTORY])
    data[0]['simulation_time'][2:]+=.1
    gap=MultiViewPACEDataset(index,'train',alpha=.5)[1]
    np.testing.assert_allclose(gap[PHASE],1.4375)
    np.testing.assert_array_equal(gap[HISTORY][:,0],[1,1,1,2])


def test_common_act_initialization_masks_fourth_view_gradients_and_roundtrip(tmp_path):
    torch.manual_seed(123)
    cfg=config();cfg.steps=10
    act_cfg=MultiViewConfig(**{k:v for k,v in asdict(cfg).items() if k not in ('phase_alpha','tactile_history','correction_horizon')})
    act_cfg.policy_type='act'
    act=MultiViewACT(act_cfg,stats(),initialize_backbone=False)
    model=MultiViewPACE(cfg,stats(),initialize_backbone=False)
    initialize_pace_from_act(model,act.state_dict())
    for key,value in act.state_dict().items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    b=pace_batch();model.eval()
    loss,_=model(b);changed=dict(b);changed['action']=b['action'].clone();changed['action'][:,10:]+=100
    torch.testing.assert_close(model(changed)[0],loss,rtol=0,atol=0)
    optimizer=torch.optim.AdamW(model.policy.get_optim_params(),lr=1e-4)
    from act_multiview import FOUR_CAMERAS
    b[FOUR_CAMERAS[-1]].requires_grad_(True)
    for _ in range(2):
        optimizer.zero_grad();b[FOUR_CAMERAS[-1]].grad=None
        loss,_=model(b);loss.backward();optimizer.step()
    assert b[FOUR_CAMERAS[-1]].grad.norm()>0
    assert all(np.isfinite(v) and v>0 for v in gradient_metrics(model).values())
    model.eval();expected=model.predict_action_chunk(b)
    path=tmp_path/'last.pt';cursor=NS(state_dict=lambda:{'epoch':0,'samples_consumed':4,'batches_consumed':2})
    index={'stats':stats(),'task':{'target_module':'nic_card_mount_0'},'index_content_sha256':'fake'}
    checkpoint(path,model,optimizer,torch.amp.GradScaler('cuda',enabled=False),cfg,index,2,cursor,1.,.01,{'loss':1.},2,True,True)
    loaded,payload=load_checkpoint(path)
    assert payload['format']==FORMAT and payload['execution']['event_before_temporal_ensemble']
    torch.testing.assert_close(loaded.predict_action_chunk(b),expected,rtol=0,atol=0)
    with patch.object(sys,'argv',['train','--resume',str(path)]):
        _,resumed,_=parse_args(expected_policy='pace')
    assert resumed.tactile_history==4
    with patch.object(sys,'argv',['train','--resume',str(path),'--phase-alpha','.2']):
        with pytest.raises(SystemExit):parse_args(expected_policy='pace')


def test_shared_snapshot_is_prepared_once_and_binds_initial_checkpoint(tmp_path,monkeypatch):
    from test_human_dataset import parent_index
    import prepare_act_pace_comparison as comparison
    monkeypatch.setattr(comparison,'DATA_DRIVE_ROOT',tmp_path)
    source=parent_index(tmp_path/'source')
    initial={'stats':source['stats'],'data_index_sha256':source['index_content_sha256'],'_initialization_sha256':'a'*64}
    source['normalization_parent_index_sha256']=initial['data_index_sha256']
    prepare=Mock(return_value=source);monkeypatch.setattr(comparison,'prepare_expert_only',prepare)
    act=MultiViewConfig(dataset_root=source['dataset_root'],init_from='same-parent.pt',eval_samples=1000,
        comparison_index_path=str(tmp_path/'shared/index.json'),expert_only=True,retain_model_motion=True)
    pace=MultiViewPACEConfig(**{**asdict(act),'policy_type':'pace'})
    first=comparison.prepare_comparison(act,initial)
    second=comparison.prepare_comparison(pace,initial)
    assert prepare.call_count==1
    assert first['index_content_sha256']==second['index_content_sha256']
    assert first['validation_anchors']==second['validation_anchors']
    report=comparison.comparison_readiness(act)
    assert report['frozen_snapshot'] and report['accepted_episodes']==26 and report['ready']
    changed=dict(initial,_initialization_sha256='b'*64)
    with pytest.raises(ValueError,match='exact same initialization'):comparison.prepare_comparison(pace,changed)
