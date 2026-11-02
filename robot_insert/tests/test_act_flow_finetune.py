"""Warm-start both trained branches on a new split, without resetting Flow."""
from dataclasses import asdict,fields
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from alternating_residual_flow_multiview import (AlternatingResidualFlowConfig,AlternatingResidualFlow,
    initialize_alternating_from_act)
from residual_flow_multiview import FORMAT,ALTERNATING_FORMAT,bind_base_act,load_residual_index
from train_sfp0_act_multiview import load_initialization
import train_sfp0_alternating_residual_flow_multiview as trainer
from test_flow_only_finetune import setup
from test_residual_flow import residual_batch
from test_act_multiview import stats
from test_fresh_act_flow import collection


@pytest.fixture(scope='module',autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('source_format',[FORMAT,ALTERNATING_FORMAT])
def test_pair_weights_and_predictions_survive_then_both_branches_update(tmp_path,source_format):
    alternating,residual,source_config,original,_=setup()
    if source_format==FORMAT:
        source=residual.eval()
        payload=dict(original,format=FORMAT,model=source.state_dict(),config=asdict(source_config),step=3500)
    else:
        source=alternating.eval();payload=original
    path=tmp_path/'source.pt';torch.save(payload,path)
    loaded,provenance=load_initialization(path,expected_policy='alternating_residual_flow')
    values={f.name:getattr(source_config,f.name) for f in fields(type(source_config))}
    values.update(policy_type='alternating_residual_flow',from_scratch=False,init_from=str(path))
    cfg=AlternatingResidualFlowConfig(**values)
    bind_base_act(cfg,loaded)
    model=AlternatingResidualFlow(cfg,stats(),initialize_backbone=False)
    report=initialize_alternating_from_act(model,loaded)
    assert report['flow_weights_preserved'] and not report['act_frozen']
    assert report['optimizer_restarted'] and len(provenance['sha256'])==64
    assert int(model.act_update_count)==int(model.flow_update_count)==0
    for key,value in model.state_dict().items():
        if key not in ('act_update_count','flow_update_count'):
            torch.testing.assert_close(value,payload['model'][key],rtol=0,atol=0)
    b=residual_batch();noise=torch.randn(2,20,7)
    model.eval()
    torch.testing.assert_close(model.predict_action_chunk(b,noise=noise),source.predict_action_chunk(b,noise=noise),
        rtol=1e-5,atol=1e-7)
    act_before=model.base_act.policy.model.action_head.weight.detach().clone()
    flow_before=model.policy.model.action_head.weight.detach().clone()
    opts,scalers=trainer.make_optimizers(model,cfg),trainer.make_scalers(cfg)
    metrics=trainer.alternating_update(model,[b],opts,scalers,cfg)
    assert metrics['act_loss']>0 and metrics['flow_loss']>0
    assert not torch.equal(act_before,model.base_act.policy.model.action_head.weight)
    assert not torch.equal(flow_before,model.policy.model.action_head.weight)
    assert int(model.act_update_count)==int(model.flow_update_count)==1


def test_new_split_preserves_normalization_and_ordinary_resume_is_still_strict(tmp_path,monkeypatch):
    root,_,_=collection(tmp_path,monkeypatch)
    initial={'stats':stats(),'data_index_sha256':'a'*64}
    cfg=AlternatingResidualFlowConfig(dataset_root=str(root),device='cpu',amp='none',
        from_scratch=False,init_from='source.pt',pretrained_backbone=False,expert_only=True,
        index_path='auto',eval_samples=10,seed=43)
    with patch.object(trainer,'load_residual_index',side_effect=AssertionError('No duplicate full cache audit')):
        index=trainer.prepare_training_index(cfg,initial=initial)
    assert index['stats']==initial['stats']
    assert index['normalization_parent_index_sha256']=='a'*64
    assert index['index_content_sha256']!=initial['data_index_sha256']
    assert len(index['episodes'])==6 and len(index['validation_anchors'])==10
    # A new run starts optimizer/cursor/BEST at zero; subsequent resume keeps the new split.
    with pytest.raises(ValueError,match='same snapshot'):load_residual_index(cfg,initial=initial)
    reference={'stats':index['stats'],'data_index_sha256':index['index_content_sha256']}
    with patch('prepare_expert_only.prepare_expert_only',side_effect=AssertionError('No repartition on resume')):
        assert trainer.prepare_training_index(cfg,resume=reference)==index


def test_finetune_entrypoint_selects_existing_pair_and_all_data_without_scratch():
    path=Path(__file__).resolve().parents[1]/'configs/sfp0_act_flow_finetune_v2.json'
    with patch.object(sys,'argv',['train','--check-data']):
        _,cfg,resume=trainer.parse_args(path)
    assert not cfg.from_scratch and cfg.index_path=='auto' and resume is None
    assert cfg.init_from.endswith('aic_sfp0_flow_only_after_scratch_v1/checkpoints/best.pt')
    assert cfg.eval_samples==1500 and cfg.eval_every==cfg.save_every==500
    assert cfg.output_dir.endswith('aic_sfp0_act_flow_finetune_v2')
