"""The unattended job must use new weights, exact controller data and resumable stages."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_direct100_pipeline as pipeline
from act_multiview import MultiViewConfig
from residual_flow_multiview import ResidualFlowConfig


def test_new_stage_configs_start_fresh_and_flow_has_no_transferred_encoder(tmp_path):
    act=pipeline.stage_config('act',tmp_path)
    flow=pipeline.stage_config('flow',tmp_path)
    assert act['init_from'] is None and not act['pretrained_backbone'] and act['steps']==10000
    assert flow['steps']==20000 and not flow['pretrained_backbone'] and not flow['flow_encoder_init_from_act']
    assert flow['init_from']==str(tmp_path/'frozen_act/act.pt')
    assert act['eval_samples']==flow['eval_samples']==1500
    assert act['eval_every']==flow['eval_every']==500
    MultiViewConfig(**dict(act,device='cpu',amp='none')).validate()
    ResidualFlowConfig(**dict(flow,device='cpu',amp='none')).validate()


def test_successful_fresh_training_is_checked_before_next_stage(tmp_path,monkeypatch):
    job=tmp_path/'job';job.mkdir();root=tmp_path/'act';monkeypatch.setattr(pipeline,'ACT',root)
    config=pipeline.stage_config('act',job);config['output_dir']=str(root);config['steps']=10
    monkeypatch.setattr(pipeline,'stage_config',lambda *a:config)
    p=pipeline.Pipeline(job);monkeypatch.setattr(p,'tick',lambda **k:None)
    calls=[]
    def run(command,log):
        calls.append(command)
        assert '--from-scratch' in command and '--resume' not in command
        directory=root/'checkpoints';directory.mkdir(parents=True)
        payload={'smoke_test':False,'config':config,'step':10,'validation_step':10,
                 'initialization':{'mode':'from_scratch'},'metrics':{'samples':1500}}
        torch.save(payload,directory/'last.pt');torch.save(payload,directory/'best.pt')
        return 0
    monkeypatch.setattr(p,'run',run)
    p.train('act');assert len(calls)==1 and p.state['stages']['act']['complete']
    p.train('act');assert len(calls)==1


def test_reject_non_scratch_checkpoint_and_preserve_old_incomplete_start(tmp_path,monkeypatch):
    job=tmp_path/'job';job.mkdir();root=tmp_path/'act';root.mkdir()
    (root/'metrics.jsonl').write_text('uncommitted old attempt')
    monkeypatch.setattr(pipeline,'ACT',root)
    config=pipeline.stage_config('act',job);config['output_dir']=str(root);config['steps']=10
    monkeypatch.setattr(pipeline,'stage_config',lambda *a:config)
    p=pipeline.Pipeline(job);monkeypatch.setattr(p,'tick',lambda **k:None)
    def run(command,log):
        assert list(tmp_path.glob('act.interrupted_*'))
        directory=root/'checkpoints';directory.mkdir(parents=True)
        payload={'smoke_test':False,'config':config,'step':10,'validation_step':10,
                 'initialization':{'mode':'from_scratch'},'metrics':{'samples':1500}}
        torch.save(payload,directory/'last.pt');torch.save(payload,directory/'best.pt')
        return 0
    monkeypatch.setattr(p,'run',run)
    p.train('act')
    payload=torch.load(root/'checkpoints/last.pt',weights_only=True)
    payload['initialization']={'mode':'fine_tune'};torch.save(payload,root/'checkpoints/last.pt')
    with pytest.raises(ValueError,match='true random'):p.checkpoint_status(root,10,'act')


def test_stop_file_also_stops_the_initial_adopted_collector(tmp_path,monkeypatch):
    job=tmp_path/'job';job.mkdir();pipeline.write(job/'initial_collection.json',{'pid':12345})
    pipeline.write(job/'STOP',{})
    p=pipeline.Pipeline(job)
    monkeypatch.setattr(p,'phase',lambda _:None)
    monkeypatch.setattr(p,'tick',lambda **_:(_ for _ in ()).throw(InterruptedError('STOP')))
    monkeypatch.setattr(pipeline,'alive',lambda *a: not signals)
    signals=[]
    monkeypatch.setattr(pipeline.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(InterruptedError):p.collect()
    assert signals==[(12345,pipeline.signal.SIGINT)]
