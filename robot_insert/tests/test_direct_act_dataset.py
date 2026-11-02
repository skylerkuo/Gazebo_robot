"""Curated direct demonstrations exclude recovery and remain portable/idempotent."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_sfp0_direct_act as prepare
import pyarrow.parquet as pq
from act_multiview import FORMAT,FOUR_CAMERAS,file_digest
from collect_sfp0_direct import import_direct_run
from sideview_data import json_write
from test_direct_collection import existing_assisted,direct_fixture


def test_curated_folder_excludes_recovery_preserves_ids_and_originals(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    monkeypatch.setattr(prepare,'DATA_DRIVE',tmp_path)
    source=tmp_path/'source';existing_assisted(source)
    for number in range(2):
        run=direct_fixture(tmp_path/f'new{number}')
        unique=run.with_name(f'direct_{number}');run.rename(unique)
        entry=import_direct_run(source,unique)[0];assert entry['accepted']
        ep=entry['episode_index'];cache=source/'_training/act_corrections/cache';cache.mkdir(parents=True,exist_ok=True)
        item={'source_sha256':entry['validation']['parquet_sha256'],'command_provenance':True,
            'source_parquet':str(source/entry['parquet'])}
        for kind,suffix in [('images','images'),('numeric','npz')]:
            path=cache/f'episode-{ep:03d}.{suffix}';path.write_bytes(f'{kind}{ep}'.encode())
            item[kind]={'path':f'cache/{path.name}','bytes':path.stat().st_size,'sha256':file_digest(path)}
        json_write(cache/f'episode-{ep:03d}.json',item)
    before=(source/'_collection/manifest.json').read_bytes()
    target=tmp_path/'direct';report=prepare.curate(source,target)
    assert report['episodes']==2 and report['episode_ids']==[1,2]
    manifest=json.loads((target/'_collection/manifest.json').read_text())
    assert len(manifest['attempts'])==2
    for entry in manifest['attempts']:
        ep=entry['episode_index']
        assert entry['recorder']['controller']=='official_cheatcode_only'
        assert set(pq.read_table(target/entry['parquet'],columns=['episode_index']).column(0).to_pylist())=={ep}
        assert (target/entry['parquet']).samefile(source/entry['parquet'])
        assert (Path(entry['run'])/'raw/trial_1_calibration.json').is_file()
        metadata=json.loads((target/f'_training/act_corrections/cache/episode-{ep:03d}.json').read_text())
        assert metadata['source_parquet']==str(target/entry['parquet'])
    assert not (target/'data/chunk-000/file-000.parquet').exists()
    assert prepare.curate(source,target)==report
    assert (source/'_collection/manifest.json').read_bytes()==before


def test_initialization_is_frozen_even_if_current_best_is_replaced(tmp_path):
    source=tmp_path/'best_act.pt';target=tmp_path/'seed/act.pt'
    def payload(step):
        return {'format':FORMAT,'camera_order':list(FOUR_CAMERAS),'step':step,
            'data_index_sha256':'a'*64,'model':{'weight':torch.tensor([step])}}
    torch.save(payload(5000),source)
    first=prepare.freeze_act_seed(source,target)
    later=tmp_path/'later.pt';torch.save(payload(6000),later);later.replace(source)
    assert prepare.freeze_act_seed(source,target)==first
    assert torch.load(target,weights_only=True)['step']==5000


def test_newly_collected_direct_episodes_prepare_their_missing_cache(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    monkeypatch.setattr(prepare,'DATA_DRIVE',tmp_path)
    source=tmp_path/'source';existing_assisted(source)
    for number in range(2):
        run=direct_fixture(tmp_path/f'fresh{number}')
        unique=run.with_name(f'direct_{number}');run.rename(unique)
        assert import_direct_run(source,unique)[0]['accepted']
    assert not (source/'_training/act_corrections/cache').exists()
    result=prepare.curate(source,tmp_path/'curated')
    assert result['episodes']==2
    for ep in result['episode_ids']:
        record=json.loads((source/f'_training/act_corrections/cache/episode-{ep:03d}.json').read_text())
        assert record['command_provenance']
        assert (tmp_path/f'curated/_training/act_corrections/cache/episode-{ep:03d}.images').is_file()


def test_direct_act_config_trains_only_act_on_separate_good_dataset():
    from train_sfp0_act_multiview import parse_args
    config=Path(__file__).resolve().parents[1]/'configs/sfp0_act_direct_success.json'
    with patch.object(sys,'argv',['train','--config',str(config),'--check-data']):
        _,cfg,_=parse_args(expected_policy='act')
    assert cfg.policy_type=='act' and cfg.expert_only and not cfg.retain_model_motion
    assert cfg.dataset_root.endswith('aic_sfp0_direct_success_v1')
    assert cfg.init_from is None and not cfg.pretrained_backbone
    assert cfg.output_dir.endswith('aic_sfp0_act_direct_scratch_v1')
    assert cfg.eval_samples==1500 and cfg.eval_every==cfg.save_every==500


def test_explicit_scratch_rejects_checkpoint_and_resume_before_loading():
    from train_sfp0_act_multiview import parse_args
    for extra in (['--init-from','old.pt'],['--resume','old.pt']):
        with patch.object(sys,'argv',['train','--from-scratch',*extra]), \
             patch('torch.load',side_effect=AssertionError('No checkpoint read')):
            with pytest.raises(SystemExit):parse_args(expected_policy='act')


def test_scratch_act_and_resnet_construct_without_any_pretrained_weights():
    from act_multiview import MultiViewACT,MultiViewConfig
    from test_act_multiview import stats
    config=MultiViewConfig(device='cpu',amp='none',pretrained_backbone=False,
        dim_model=32,n_heads=4,dim_feedforward=64,n_encoder_layers=1,n_decoder_layers=1)
    with patch('torch.load',side_effect=AssertionError('No checkpoint read')), \
         patch('torch.hub.load_state_dict_from_url',side_effect=AssertionError('No ImageNet download')):
        model=MultiViewACT(config,stats())
    assert model.policy.config.pretrained_backbone_weights is None
    assert all(p.requires_grad for p in model.policy.model.backbone.parameters())


def test_scratch_expert_act_prepares_new_training_statistics(tmp_path,monkeypatch):
    from test_fresh_act_flow import collection
    from act_multiview import MultiViewConfig
    from train_sfp0_act_multiview import load_training_index
    root,_,_=collection(tmp_path,monkeypatch)
    config=MultiViewConfig(dataset_root=str(root),index_path='auto',device='cpu',amp='none',
        init_from=None,pretrained_backbone=False,expert_only=True,eval_samples=10,seed=43)
    index=load_training_index(config)
    assert index['normalization_source']=='current training episodes only'
    assert 'normalization_parent_index_sha256' not in index
    assert config.index_path.endswith('/index.json')
