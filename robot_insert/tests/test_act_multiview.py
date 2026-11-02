"""Four-view ACT: automatic grouped 5% VAL split, causal labels, physical metrics and resume."""
import copy
from contextlib import redirect_stderr
from dataclasses import asdict
import io
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ftact import STATE,WRENCH,FTNormalizer,load_checkpoint
from act_multiview import (FORMAT,FOUR_CAMERAS,MultiViewACT,MultiViewConfig,MultiViewDataset,
    split_episodes,extended_split,validation_anchors,training_statistics,index_digest,read_index,INDEX_FORMAT,file_digest)
from act_objectives import common_scale_loss,evaluate_pose_chunks
from train_sfp0_act_multiview import checkpoint,parse_args,load_training_index,set_optimizer_learning_rates
from train_sfp0_act import lr_factor
from train_ftact import restore_rng


def stats():
    value={k:{'mean':[0.]*6,'std':[1.]*6} for k in (STATE,WRENCH)}
    value['action']={'mean':[.4,-.1,.2,0.,0.,0.,1.],'std':[.01,.01,.08,.01,.01,.01,.06]}
    value.update({k:{'mean':[[[.485]],[[.456]],[[.406]]],
        'std':[[[.229]],[[.224]],[[.225]]]} for k in FOUR_CAMERAS})
    return value


def config():
    return MultiViewConfig(device='cpu',amp='none',dim_model=32,n_heads=4,dim_feedforward=64,
        n_encoder_layers=1,n_decoder_layers=1,pretrained_backbone=False,dropout=.1,batch_size=2)


def batch():
    return {STATE:torch.randn(2,6),WRENCH:torch.randn(2,6),
        **{k:torch.rand(2,3,256,288) for k in FOUR_CAMERAS},
        'action':torch.tensor(stats()['action']['mean']).repeat(2,20,1)+torch.randn(2,20,7)*.001,
        'action_is_pad':torch.zeros(2,20,dtype=torch.bool)}


@pytest.fixture(scope='module',autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_100_episodes_are_exactly_95_train_5_val_and_no_frame_split():
    train,val=split_episodes(list(range(100)))
    assert len(train)==95 and len(val)==5
    assert not set(train)&set(val) and sorted(train+val)==list(range(100))
    assert (train,val)==split_episodes(list(reversed(range(100))))
    assert (train,val)!=split_episodes(list(range(100)),43)
    with pytest.raises(ValueError):split_episodes([0,0])


def test_1000_validation_anchors_are_fixed_unique_balanced_and_future_safe():
    episodes=[{'episode_index':e,'length':400+e} for e in (4,8,16,32,64)]
    anchors=validation_anchors(episodes)
    assert len(anchors)==len(set(map(tuple,anchors)))==1000
    assert anchors==validation_anchors(episodes)
    assert all(sum(ep==e['episode_index'] for ep,f in anchors)==200 for e in episodes)
    lengths={e['episode_index']:e['length'] for e in episodes}
    assert all(0<=f and f+20<lengths[ep] for ep,f in anchors)
    with pytest.raises(ValueError):validation_anchors(episodes[:1])


def test_statistics_only_include_supplied_training_episodes_and_use_floors():
    train=[{STATE:np.zeros((4,6)),WRENCH:np.ones((4,6)),
            'action':np.tile([0,0,.2,0,0,0,1.],(4,1))}]
    value=training_statistics(train)
    assert value[STATE]['mean']==[0.]*6 and value[STATE]['count']==[4]
    assert value[STATE]['std']==[.001]*6
    assert value[WRENCH]['std']==[.5]*3+[.05]*3
    assert value['action']['std']==[.01]*7
    assert all(k in value for k in FOUR_CAMERAS)


def test_equal_mm_xyz_error_has_equal_loss_and_gradient_and_matches_latest_method():
    from pace_act import common_scale_loss as previous_loss
    normalizer=FTNormalizer(stats(),FOUR_CAMERAS)
    target=torch.tensor(stats()['action']['mean']).view(1,1,7)
    losses,gradients=[],[]
    for axis in range(3):
        physical=target.clone();physical[...,axis]+=.001;physical.requires_grad_()
        loss,metrics=common_scale_loss(normalizer.actions(physical),target,normalizer)
        old,old_metrics=previous_loss(normalizer.actions(physical),target,normalizer)
        torch.testing.assert_close(loss,old,rtol=0,atol=0)
        loss.backward();losses.append(loss.item());gradients.append(physical.grad[...,axis].item())
    np.testing.assert_allclose(losses,[.1/7]*3,rtol=3e-5)
    np.testing.assert_allclose(gradients,[1/.07]*3,rtol=1e-6)


@pytest.mark.parametrize('sample_count',[1000,1500,2000])
def test_validation_reports_xyz_mm_bias_cm_horizons_and_preserves_mode(sample_count):
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.normalizer=FTNormalizer(stats(),FOUR_CAMERAS)
        def normalized_action_chunk(self,b):
            target=b['action'].clone();target[...,:3]+=torch.tensor([.001,-.002,.010])
            target[...,3:]*=-1
            return self.normalizer.actions(target)
    model=Model().eval();target=torch.tensor(stats()['action']['mean']).repeat(400,20,1)
    cfg=config();cfg.eval_samples=sample_count
    batches=[{'action':target} for _ in range(sample_count//400)]
    if sample_count%400:batches.append({'action':target[:sample_count%400]})
    result=evaluate_pose_chunks(model,batches,cfg)
    np.testing.assert_allclose(result['position_xyz_mae_mm'],[1,2,10],atol=3e-5)
    np.testing.assert_allclose(result['position_xyz_bias_mm'],[1,-2,10],atol=3e-5)
    assert result['position_l2_mm']==pytest.approx(np.sqrt(105),abs=3e-5)
    assert result['position_l2_cm']==pytest.approx(np.sqrt(105)/10,abs=3e-6)
    assert result['rotation_deg']==pytest.approx(0,abs=1e-8)
    assert len(result['position_xyz_by_horizon_mm'])==20
    assert result['samples']==sample_count and result['valid_action_steps']==sample_count*20
    assert not model.training
    with pytest.raises(ValueError,match=str(sample_count)):evaluate_pose_chunks(model,[{'action':target}],cfg)


def test_packed_dataset_has_four_rgb_views_and_never_crosses_episode(tmp_path):
    entries=[]
    for ep in range(2):
        blob=tmp_path/f'{ep}.images';offsets=np.empty((25,4,2),dtype=np.int64)
        with blob.open('wb') as stream:
            for frame in range(25):
                for camera in range(4):
                    bgr=np.full((256,288,3),(ep*100,camera*50,frame*4),dtype=np.uint8)
                    png=cv2.imencode('.png',bgr)[1].tobytes()
                    offsets[frame,camera]=[stream.tell(),len(png)];stream.write(png)
        action=np.zeros((25,7),dtype=np.float32);action[:,2]=np.arange(25)+ep*100;action[:,6]=1
        arrays=tmp_path/f'{ep}.npz'
        np.savez(arrays,**{STATE:np.full((25,6),ep,dtype=np.float32),
            WRENCH:np.full((25,6),ep+1,dtype=np.float32),'action':action,'image_offsets':offsets})
        entries.append({'episode_index':ep,'length':25,'split':'train' if ep==0 else 'validation',
                        'images':{'path':blob.name},'numeric':{'path':arrays.name}})
    index={'_directory':str(tmp_path),'episodes':entries,'validation_anchors':[[1,3]]}
    train=MultiViewDataset(index,'train');val=MultiViewDataset(index,'validation')
    assert len(train)==5 and len(val)==1
    first,last=train[0],train[-1]
    assert first['action'][:,2].tolist()==list(range(1,21))
    assert last['action'][:,2].tolist()==list(range(5,25))
    assert val[0]['action'][:,2].tolist()==list(range(104,124))
    side=first[FOUR_CAMERAS[-1]]
    torch.testing.assert_close(side[:,0,0],torch.tensor([0.,150/255,0.]))
    copy=pickle.loads(pickle.dumps(train))
    torch.testing.assert_close(copy[0][FOUR_CAMERAS[-1]],side)
    assert not first['action_is_pad'].any()
    train.close();val.close();copy.close()


def test_model_uses_side_image_force_and_backbone_without_pace_or_answer_input():
    torch.manual_seed(42)
    model=MultiViewACT(config(),stats(),initialize_backbone=False)
    assert list(model.policy.config.image_features)==list(FOUR_CAMERAS)
    assert not model.policy.config.use_vae
    assert not any('pace' in name.lower() for name,_ in model.named_parameters())
    b=batch();b[FOUR_CAMERAS[-1]].requires_grad_()
    loss,_=model(b);loss.backward()
    assert b[FOUR_CAMERAS[-1]].grad.norm()>0
    assert model.policy.model.encoder_robot_state_input_proj.weight.grad[:,6:].norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.policy.model.backbone.parameters())
    optimized={id(p) for g in model.policy.get_optim_params() for p in g['params']}
    assert optimized=={id(p) for p in model.parameters() if p.requires_grad}
    model.eval()
    with torch.no_grad():
        before=model.normalized_action_chunk(b)
        b['action']=torch.full_like(b['action'],1000.)
        torch.testing.assert_close(before,model.normalized_action_chunk(b),rtol=0,atol=0)
        pose=model.predict_action_chunk(b)
        torch.testing.assert_close(pose[...,3:].norm(dim=-1),torch.ones(2,20))
    del b[FOUR_CAMERAS[-1]]
    with pytest.raises(KeyError):model.normalized_action_chunk(b)


def test_checkpoint_optimizer_rng_resume_matches_next_update(tmp_path):
    torch.manual_seed(52)
    cfg=config();model=MultiViewACT(cfg,stats(),initialize_backbone=False).train()
    optimizer=torch.optim.AdamW(model.policy.get_optim_params(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    b=batch()
    def update(m,o):
        o.zero_grad(set_to_none=True);loss,_=m(b);loss.backward();o.step();return loss.detach()
    update(model,optimizer)
    index={'stats':stats(),'task':{'target_module':'nic_card_mount_0'},'index_content_sha256':'frozen'}
    cursor=SimpleNamespace(state_dict=lambda:{'epoch':0,'batches_consumed':1,'samples_consumed':2})
    path=tmp_path/'last.pt'
    checkpoint(path,model,optimizer,scaler,cfg,index,1,cursor,.2,.01,None,None,False,True)
    expected=update(model,optimizer)
    restored,payload=load_checkpoint(path,'cpu');restored.train()
    resumed_optimizer=torch.optim.AdamW(restored.policy.get_optim_params(),lr=1e-4)
    resumed_optimizer.load_state_dict(payload['optimizer']);restore_rng(payload['rng'])
    actual=update(restored,resumed_optimizer)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    assert payload['validation_step'] is None and payload['camera_order']==list(FOUR_CAMERAS)


def test_resume_rejects_changed_objective_index_output_or_legacy_weights(tmp_path):
    cfg=config()
    payload={'format':FORMAT,'optimizer':{},'config':asdict(cfg),'step':1}
    p=tmp_path/'last.pt';torch.save(payload,p)
    for options in (['--index-path','/mnt/d/other/index.json'],['--output-dir','/mnt/d/other'],['--steps','1']):
        with patch.object(sys,'argv',['train','--resume',str(p),*options]),redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit):parse_args()
    payload['format']='aic-sfp0-act-v2';torch.save(payload,p)
    with patch.object(sys,'argv',['train','--resume',str(p)]),redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):parse_args()


def test_resume_keeps_old_rates_unless_explicitly_overridden_and_persists_override(tmp_path):
    cfg=config();cfg.learning_rate=1e-4;cfg.backbone_lr=1e-5
    payload={'format':FORMAT,'optimizer':{},'config':asdict(cfg),'step':9020}
    p=tmp_path/'last.pt';torch.save(payload,p)
    with patch.object(sys,'argv',['train','--resume',str(p)]):
        _,unchanged,_=parse_args()
    assert unchanged.learning_rate==1e-4 and unchanged.backbone_lr==1e-5
    with patch.object(sys,'argv',['train','--resume',str(p),'--learning-rate','3e-5','--backbone-lr','3e-6']):
        _,lower,resume= parse_args()
    assert lower.learning_rate==3e-5 and lower.backbone_lr==3e-6
    assert lower.index_path==cfg.index_path and lower.steps==cfg.steps
    assert lower.warmup_steps==cfg.warmup_steps and resume['step']==9020
    payload['config']=asdict(lower);torch.save(payload,p)
    with patch.object(sys,'argv',['train','--resume',str(p)]):
        _,continued,_=parse_args()
    assert continued==lower


def test_lower_lr_applies_after_optimizer_load_without_resetting_momentum_or_schedule(tmp_path):
    cfg=config();cfg.learning_rate=1e-4;cfg.backbone_lr=1e-5
    params=[torch.nn.Parameter(torch.tensor([1.])),torch.nn.Parameter(torch.tensor([2.]))]
    optimizer=torch.optim.AdamW([{'params':[params[0]],'lr':1e-4},{'params':[params[1]],'lr':1e-5}])
    for p in params:p.grad=torch.ones_like(p)
    optimizer.step()
    saved=copy.deepcopy(optimizer.state_dict())
    optimizer=torch.optim.AdamW([{'params':[params[0]],'lr':3e-5},{'params':[params[1]],'lr':3e-6}])
    optimizer.load_state_dict(saved)  # This restores the old LR too.
    assert optimizer.param_groups[0]['lr']==1e-4
    cfg.learning_rate=3e-5;cfg.backbone_lr=3e-6
    set_optimizer_learning_rates(optimizer,cfg,9021)
    assert optimizer.param_groups[0]['lr']==pytest.approx(2.4813070530897445e-5,rel=1e-4)
    assert optimizer.param_groups[1]['lr']==pytest.approx(optimizer.param_groups[0]['lr']/10)
    for state,prior in zip(optimizer.state.values(),saved['state'].values()):
        for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(state[key],prior[key],rtol=0,atol=0)
    optimizer.step()
    assert all(state['step'].item()==2 for state in optimizer.state.values())
    p=tmp_path/'optimizer.pt';torch.save(optimizer.state_dict(),p)
    stored=torch.load(p,weights_only=True)
    assert stored['param_groups'][0]['lr']==optimizer.param_groups[0]['lr']
    # An ordinary resume still produces the same scheduled rates as before this change.
    cfg.learning_rate=1e-4;cfg.backbone_lr=1e-5
    set_optimizer_learning_rates(optimizer,cfg,9021)
    assert optimizer.param_groups[0]['lr']==cfg.learning_rate*lr_factor(9021,cfg)


@pytest.mark.parametrize('key',['learning_rate','backbone_lr'])
@pytest.mark.parametrize('value',[0.,-1.,float('inf'),float('nan')])
def test_invalid_learning_rates_rejected(key,value):
    cfg=config();setattr(cfg,key,value)
    with pytest.raises(ValueError,match=key):cfg.validate()


def source_manifest(count):
    return {'attempts':[{'accepted':True,'episode_index':i,'attempt':i+5,
        'official_score':{'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}},
        'recorder':{'teacher_return':True,'writer_error':None},
        'validation':{'frames':640,'parquet_sha256':str(i).zfill(64)}} for i in range(count)]}


@pytest.mark.parametrize('count,expected_val',[(26,2),(50,3),(56,3),(100,5),(101,6)])
def test_any_available_episode_count_uses_rounded_up_five_percent(count,expected_val):
    from prepare_sfp0_act_multiview import select_source_episodes,snapshot_settings
    manifest=source_manifest(count)
    manifest['attempts'] += [{'accepted':False,'attempt':1000},{'attempt':1001}]
    selected=select_source_episodes(manifest)
    train,val=split_episodes([e['episode_index'] for e in selected])
    assert len(selected)==count and len(val)==expected_val and len(train)==count-expected_val
    assert not set(train)&set(val)
    settings=snapshot_settings('/mnt/d/source',selected,42)
    assert settings['episodes']==count and settings['validation_samples']==1000
    assert settings['validation_fraction']==.05
    assert len(validation_anchors([{'episode_index':e,'length':640} for e in val]))==1000


def test_snapshot_identity_changes_when_new_accepted_episodes_arrive():
    from prepare_sfp0_act_multiview import select_source_episodes,snapshot_settings
    manifest=source_manifest(50)
    selected=select_source_episodes(manifest)
    frozen=snapshot_settings('/mnt/d/source',selected,42)
    larger=source_manifest(56)
    expanded=snapshot_settings('/mnt/d/source',select_source_episodes(larger),42)
    assert index_digest(frozen)!=index_digest(expanded)
    assert frozen==snapshot_settings('/mnt/d/source',select_source_episodes(larger,episodes=50),42)
    assert len(selected)==50
    with pytest.raises(ValueError,match='only 50'):select_source_episodes(manifest,episodes=100)
    with pytest.raises(ValueError,match='at least two'):select_source_episodes(source_manifest(1))


def test_extended_split_to_210_never_places_pretrained_training_episodes_in_val():
    old_train,old_val=split_episodes(list(range(74)))
    sources=[[i,i+5,str(i).zfill(64)] for i in range(210)]
    parent={'preparation_settings':{'sources':sources[:74]},
        'episodes':[{'episode_index':i,'split':'train' if i in old_train else 'validation'} for i in range(74)]}
    train,val=extended_split(sources,parent,42)
    assert len(train)==199 and len(val)==11
    assert set(range(74))<=set(train)
    assert not set(old_train)&set(val)
    assert len(val)==11 and min(val)>=74
    assert (train,val)==extended_split(sources,parent,42)
    changed=copy.deepcopy(sources);changed[0][2]='changed'
    with pytest.raises(ValueError,match='removed or changed'):extended_split(changed,parent,42)


def test_refresh_flag_requires_initialization_and_a_fresh_run(tmp_path):
    with patch.object(sys,'argv',['train','--refresh-data','--device','cpu','--amp','none']),redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):parse_args()
    with patch.object(sys,'argv',['train','--resume','missing.pt','--refresh-data']),redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):parse_args()


def test_training_auto_detects_once_then_resume_uses_the_saved_frozen_index(tmp_path):
    cfg=config()
    frozen=tmp_path/'snapshot';frozen.mkdir();(frozen/'index.json').write_text('{}')
    index={'_directory':str(frozen),'dataset_root':cfg.dataset_root,'validation_samples':1000,
           'index_content_sha256':'frozen'}
    with patch('prepare_sfp0_act_multiview.prepare',return_value=index) as prepare:
        assert load_training_index(cfg) is index
        prepare.assert_called_once_with(cfg.dataset_root,seed=cfg.seed,eval_samples=cfg.eval_samples)
        assert cfg.index_path==str(frozen/'index.json')
        with patch('train_sfp0_act_multiview.read_index',return_value=index):
            assert load_training_index(cfg,{'data_index_sha256':'frozen'}) is index
            assert prepare.call_count==1
            with pytest.raises(ValueError,match='same frozen'):
                load_training_index(cfg,{'data_index_sha256':'different'})
    (frozen/'index.json').unlink()
    with pytest.raises(FileNotFoundError,match='Frozen training index'):
        load_training_index(cfg,{'data_index_sha256':'frozen'})
    cfg.index_path='auto'
    with pytest.raises(ValueError,match='saved in its checkpoint'):
        load_training_index(cfg,{'data_index_sha256':'frozen'})


def test_cli_defaults_are_auto_1500_samples_every_500_steps():
    with patch.object(sys,'argv',['train','--device','cpu','--amp','none']):
        _,cfg,resume=parse_args()
    assert cfg.index_path=='auto' and cfg.eval_samples==1500 and cfg.eval_every==500
    assert cfg.save_every==500 and resume is None


@pytest.mark.parametrize('sample_count',[1000,1500])
def test_v2_index_accepts_non_100_episode_snapshots_and_rejects_val_leakage(tmp_path,sample_count):
    import copy
    from prepare_sfp0_act_multiview import snapshot_settings
    source=source_manifest(26)
    length=840 if sample_count==1500 else 640
    for entry in source['attempts']:entry['validation']['frames']=length
    train,val=split_episodes(list(range(26)))
    episodes=[]
    for entry in source['attempts']:
        ep=entry['episode_index']
        item={'episode_index':ep,'source_attempt':entry['attempt'],'source_sha256':entry['validation']['parquet_sha256'],
              'length':length,'split':'validation' if ep in val else 'train'}
        # read_index validates artifact size/hash; RGB/NPZ decoding is tested separately.
        for kind in ('images','numeric'):
            artifact=tmp_path/f'{ep}.{kind}'
            artifact.write_bytes(f'{ep}-{kind}'.encode())
            item[kind]={'path':artifact.name,'bytes':artifact.stat().st_size,'sha256':file_digest(artifact)}
        episodes.append(item)
    value={'format':INDEX_FORMAT,'dataset_root':'/mnt/d/source','fps':20,'action_offset':1,'chunk_size':20,
        'cameras':list(FOUR_CAMERAS),'input_shape':[3,256,288],'episodes':episodes,
        'preparation_settings':snapshot_settings('/mnt/d/source',source['attempts'],42),
        'validation_fraction':.05,'validation_samples':sample_count,
        'validation_anchors':validation_anchors([e for e in episodes if e['split']=='validation'],count=sample_count),
        'stats':{key:{'count':[len(train)*length]} for key in (STATE,WRENCH,'action')}}
    path=tmp_path/'index.json'
    def save(index):
        index['index_content_sha256']=index_digest(index)
        path.write_text(json.dumps(index))
    save(value)
    loaded=read_index(path)
    assert len(loaded['episodes'])==26 and len(loaded['validation_anchors'])==sample_count
    leaking=copy.deepcopy(value);leaking['validation_anchors'][0]=[train[0],0];save(leaking)
    with pytest.raises(ValueError,match='leaks'):read_index(path)
    duplicated=copy.deepcopy(value);duplicated['validation_anchors'][0]=duplicated['validation_anchors'][1];save(duplicated)
    with pytest.raises(ValueError,match=f'{sample_count} fixed unique'):read_index(path)
    wrong_source=copy.deepcopy(value);wrong_source['episodes'][0]['source_attempt']=999;save(wrong_source)
    with pytest.raises(ValueError,match='source membership'):read_index(path)
    wrong_count=copy.deepcopy(value);wrong_count['validation_samples']=1200;save(wrong_count)
    with pytest.raises(ValueError,match='1000, 1500 or 2000 fixed'):read_index(path)
    save(value)
    artifact=tmp_path/value['episodes'][0]['images']['path']
    artifact.write_bytes(b'x'*artifact.stat().st_size)
    with pytest.raises(ValueError,match='Changed images cache'):read_index(path)


def test_preparation_explains_when_five_percent_has_too_few_unique_val_samples(tmp_path):
    from prepare_sfp0_act_multiview import prepare
    (tmp_path/'_collection').mkdir()
    (tmp_path/'_collection/manifest.json').write_text(json.dumps(source_manifest(20)))
    with pytest.raises(ValueError,match='only 620 complete anchors; need 1000'):
        prepare(tmp_path,Path('/mnt/d/nonexistent-auto-test-output'))


def test_index_digest_covers_membership_statistics_and_camera_order():
    value={'episodes':[1,2],'stats':{'x':1},'cameras':list(FOUR_CAMERAS)}
    digest=index_digest(value);value['index_content_sha256']=digest
    assert index_digest(value)==digest
    value['cameras'].reverse()
    assert index_digest(value)!=digest
