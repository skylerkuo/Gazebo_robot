"""Successful intervention import, provenance filtering and frozen-parent fine-tuning."""
import copy
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from act_multiview import (FOUR_CAMERAS,INDEX_FORMAT,FORMAT,MultiViewConfig,MultiViewACT,MultiViewDataset,
                          file_digest,index_digest,read_index,split_episodes,training_statistics,validation_anchors)
from ftact import STATE,WRENCH
from sideview_data import CAMERAS,TASK
from sideview_writer import EpisodeWriter
from human_dataset import import_human_run
from prepare_human_corrections import prepare_corrections
from train_sfp0_act_multiview import load_initialization,load_training_index
import pyarrow.parquet as pq


def profile():
    result=json.loads((ROOT/'configs/sfp0_sideview.json').read_text())
    result.update(record_width=5,record_height=4)
    return result


def recorded_run(root,*,gap=False,expert=False,model_only=False):
    run=root/'runs/session';raw=run/'raw';raw.mkdir(parents=True)
    settings=profile();writer=EpisodeWriter(raw,'task_1',settings)
    message=NS(width=5,height=4,step=15,encoding='rgb8',data=np.arange(60,dtype=np.uint8).tobytes())
    for frame in range(65):
        stamp=1+frame*.05+(.05 if gap and frame>=5 else 0.)
        source='model' if frame<10 else ('expert' if expert else 'human') if frame<45 else 'hold'
        if model_only and source!='hold':source='model'
        segment=0 if frame<10 else 1 if frame<45 else 2
        writer.put(('command',{'simulation_time':stamp-.001,'pose':[frame*.0001,0,.2,0,0,0,1],
            'control_source':source,'control_segment':segment,'human_active':source in ('human','expert')}))
        writer.put(('frame',{'simulation_time':stamp,'camera_times':{c:stamp for c in CAMERAS},
            'joint_time':stamp-.002,'wrench_time':stamp-.002,'state':[1.,2,3,4,5,6],
            'arm_joint_positions':[1.,2,3,4,5,6],'wrench':[1.,2,3,4,5,6],
            'tcp_pose':[0,0,.2,0,0,0,1]},[message]*4))
    writer.finish()
    (raw/'task_1.json').write_text(json.dumps({'policy_return':True,'teacher_return':None,'writer_error':None,
        'frames':65,'commands':65,'dropped_frames':{},'pending_at_end':0,
        'controller':'model_and_official_expert' if expert else 'model_and_human'}))
    (raw/'task_1_calibration.json').write_text(json.dumps({'side_camera':settings['camera'],'record_width':5,'record_height':4}))
    (run/'scoring').mkdir()
    (run/'scoring/scoring.yaml').write_text(yaml.safe_dump({'trial_1':{
        'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}}}))
    return run


def parent_index(root):
    root.mkdir();(root/'_collection').mkdir()
    (root/'_collection/manifest.json').write_text(json.dumps({'settings':profile()}))
    rgb=np.resize(np.arange(255,dtype=np.uint8),(256,288,3))
    png=cv2.imencode('.png',rgb)[1].tobytes()
    (root/'base.images').write_bytes(png)
    arrays={STATE:np.tile([1.,2,3,4,5,6],(640,1)).astype(np.float32),
        WRENCH:np.tile([1.,2,3,4,5,6],(640,1)).astype(np.float32),
        'action':np.tile([0,0,.2,0,0,0,1.],(640,1)).astype(np.float32)}
    np.savez(root/'base.npz',**arrays,image_offsets=np.tile([0,len(png)],(640,4,1)))
    train,val=split_episodes(list(range(26)))
    episodes=[]
    for ep in range(26):
        e={'episode_index':ep,'source_attempt':ep,'source_sha256':str(ep).zfill(64),'length':640,
           'split':'train' if ep in train else 'validation'}
        for kind,name in [('images','base.images'),('numeric','base.npz')]:
            p=root/name;e[kind]={'path':name,'bytes':p.stat().st_size,'sha256':file_digest(p)}
        episodes.append(e)
    value={'format':INDEX_FORMAT,'dataset_root':str(root),'task':TASK,'fps':20,'chunk_size':20,'action_offset':1,
        'cameras':list(FOUR_CAMERAS),'input_shape':[3,256,288],'validation_samples':1000,'validation_fraction':.05,
        'preparation_settings':{'episodes':26,'seed':42,'sources':[[e['episode_index'],e['source_attempt'],e['source_sha256']] for e in episodes]},
        'episodes':episodes,'validation_anchors':validation_anchors([e for e in episodes if e['split']=='validation']),
        'stats':training_statistics([arrays]*len(train)),
        'counts':{'train':{'episodes':24,'frames':24*640,'anchors':24*620},
                  'validation':{'episodes':2,'frames':1280,'anchors':1240}}}
    value['index_content_sha256']=index_digest(value)
    (root/'index.json').write_text(json.dumps(value))
    return read_index(root/'index.json')


def test_import_keeps_provenance_and_only_human_chunks_train(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    monkeypatch.setattr('prepare_human_corrections.DATA_DRIVE_ROOT',tmp_path)
    base=parent_index(tmp_path/'parent');human=tmp_path/'human';run=recorded_run(human,gap=True)
    entry=import_human_run(human,run,profile(),base['dataset_root'],status=0)
    assert entry['accepted'] and entry['validation']['expert_anchor_count']==16
    assert entry['expert_anchor_frames']==list(range(9,25))
    rows=pq.read_table(human/entry['parquet']).to_pydict()
    assert rows['index']==list(range(65))
    assert set(rows['action_source'])=={'model','human','hold'}
    assert entry['validation']['maximum_frame_gap_s']==pytest.approx(.1)
    combined=prepare_corrections(base,human)
    assert combined['validation_anchors']==base['validation_anchors'] and combined['stats']==base['stats']
    assert combined['counts']['train']['human_anchors']==16
    train=MultiViewDataset(combined,'train',correction_repeat=4)
    val=MultiViewDataset(combined,'validation')
    assert len(train)==24*620+16*4 and len(val)==1000
    sample=train[-1]
    assert int(sample['episode_index'])==26 and sample['action'].shape==(20,7)
    f=entry['expert_anchor_frames'][-1]
    np.testing.assert_allclose(sample['action'][:,0],np.arange(f+1,f+21)*.0001,atol=1e-7)
    assert all(sample[key].shape==(3,256,288) for key in FOUR_CAMERAS)
    train.close();val.close()
    cfg=MultiViewConfig(dataset_root=base['dataset_root'],device='cpu',amp='none',pretrained_backbone=False,
        dim_model=32,n_heads=4,dim_feedforward=64,n_encoder_layers=1,n_decoder_layers=1)
    torch.set_num_threads(2)
    parent=MultiViewACT(cfg,base['stats'],initialize_backbone=False).eval()
    parent_path=tmp_path/'parent.pt'
    torch.save({'format':FORMAT,'camera_order':list(FOUR_CAMERAS),'step':6000,'stats':base['stats'],
        'data_index_sha256':base['index_content_sha256'],'model':parent.state_dict(),
        'config':{'index_path':str(Path(base['_directory'])/'index.json')}},parent_path)
    initial,provenance=load_initialization(parent_path)
    cfg.init_from=str(parent_path);cfg.corrections_root=str(human)
    prepared=load_training_index(cfg,initial=initial)
    child=MultiViewACT(cfg,prepared['stats'],initialize_backbone=False).eval()
    child.load_state_dict(initial['model'],strict=True)
    batch={k:v.unsqueeze(0) for k,v in sample.items()}
    with torch.no_grad():torch.testing.assert_close(parent.normalized_action_chunk(batch),child.normalized_action_chunk(batch),rtol=0,atol=0)
    child.train();loss,_=child(batch);loss.backward()
    assert torch.isfinite(loss) and child.policy.model.action_head.weight.grad.norm()>0
    assert provenance['step']==6000


def test_official_expert_import_and_finetune_keep_real_action_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    monkeypatch.setattr('prepare_human_corrections.DATA_DRIVE_ROOT',tmp_path)
    base=parent_index(tmp_path/'parent');root=tmp_path/'expert';run=recorded_run(root,expert=True)
    summary_path=run/'raw/task_1.json';summary=json.loads(summary_path.read_text())
    summary['controller_version']='in_place_progress_v2';summary_path.write_text(json.dumps(summary))
    entry=import_human_run(root,run,profile(),base['dataset_root'],status=0,expert_source='expert')
    assert entry['accepted'] and entry['validation']['control_source_counts']['expert']==35
    assert entry['controller_version']=='in_place_progress_v2'
    assert entry['validation']['expert_anchor_count']==16
    manifest=json.loads((root/'_collection/manifest.json').read_text())
    assert manifest['active_controller_version']=='in_place_progress_v2'
    assert manifest['ground_truth_supervision'] and not manifest['model_ground_truth_inputs']
    combined=prepare_corrections(base,root)
    assert combined['validation_anchors']==base['validation_anchors'] and combined['stats']==base['stats']
    correction=combined['correction_episodes'][0]
    assert correction['expert_source']=='expert'
    assert combined['counts']['train']['expert_anchors']==16 and 'human_anchors' not in combined['counts']['train']
    train=MultiViewDataset(combined,'train');sample=train[-1]
    assert sample['action'].shape==(20,7)
    f=correction['train_anchor_frames'][-1]
    np.testing.assert_allclose(sample['action'][:,0],np.arange(f+1,f+21)*.0001,atol=1e-7)
    train.close()


def test_expert_only_training_survives_deleted_parent_and_val_uses_only_expert_windows(tmp_path,monkeypatch):
    monkeypatch.setattr('prepare_expert_only.DATA_DRIVE_ROOT',tmp_path)
    base=parent_index(tmp_path/'old');root=tmp_path/'expert';root.mkdir();(root/'_collection').mkdir()
    old=Path(base['dataset_root']);shutil.copyfile(old/'base.images',root/'shared.images')
    with np.load(old/'base.npz') as a:arrays={k:a[k].copy() for k in a.files}
    arrays['action'][:,0]=np.arange(640)*.0001
    arrays.update(action_source=np.array(['model']*20+['expert']*620),action_segment=np.array([0]*20+[1]*620,dtype=np.int64),
        human_active=np.ones(640,dtype=bool),simulation_time=np.arange(640)*.05)
    np.savez(root/'shared.npz',**arrays)
    initial={'stats':base['stats'],'data_index_sha256':base['index_content_sha256'],
        'config':{'index_path':str(old/'index.json')}}
    shutil.rmtree(old)
    score={'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}}
    entries=[{'accepted':True,'episode_index':i,'attempt':i,'expert_anchor_frames':list(range(19,620)),
        'parquet':f'{i}.parquet',
        'validation':{'frames':640,'parquet_sha256':str(i).zfill(64)},'official_score':score,
        'recorder':{'policy_return':True,'writer_error':None}} for i in range(62)]
    from human_dataset import EXPERT_FORMAT
    manifest={'format':EXPERT_FORMAT,'task':TASK,'attempts':entries}
    (root/'_collection/manifest.json').write_text(json.dumps(manifest))
    def prepared(source,output,entry,threads,**kwargs):
        item={'episode_index':entry['episode_index'],'source_attempt':entry['attempt'],'length':640,
            'source_sha256':entry['validation']['parquet_sha256']}
        for kind,name in [('images','shared.images'),('numeric','shared.npz')]:
            p=root/name;item[kind]={'path':os.path.relpath(p,output),'bytes':p.stat().st_size,'sha256':file_digest(p)}
        return item
    monkeypatch.setattr('prepare_expert_only.prepare_episode',prepared)
    cfg=MultiViewConfig(dataset_root=str(root),expert_only=True,init_from='old-best.pt',device='cpu',amp='none',eval_samples=2000)
    index=load_training_index(cfg,initial=initial)
    assert len(index['validation_anchors'])==2000 and index['stats']==initial['stats']
    assert index['counts']['train']['episodes']==58 and index['counts']['validation']['episodes']==4
    train=MultiViewDataset(index,'train');val=MultiViewDataset(index,'validation')
    assert len(train)==58*601 and len(val)==2000 and val[0]['action'].shape==(20,7)
    assert load_training_index(cfg,resume={'data_index_sha256':index['index_content_sha256']})['stats']==initial['stats']
    train.close();val.close()
    from rollout_sampling import rollout_windows
    def rollout_file(_):
        weights,frames,validation,report=rollout_windows(arrays['action_source'],arrays['action_segment'],arrays['human_active'],
            arrays['simulation_time'],arrays['action'],arrays[WRENCH],arrays['action'])
        return arrays['action'].copy(),weights,frames,validation,report
    monkeypatch.setattr('rollout_sampling.sample_rollout_file',rollout_file)
    cfg.retain_model_motion=True;cfg.index_path='auto'
    mixed=load_training_index(cfg,initial=initial)
    assert mixed['format']=='aic-sfp0-act-assisted-rollout-index-v1'
    training=MultiViewDataset(mixed,'train');validation=MultiViewDataset(mixed,'validation')
    assert len(training)==58*620
    sample=training[0]
    assert sample['action_mask'].dtype==torch.bool and sample['action_mask'].all()
    assert 'action_weights' not in sample
    assert validation[0]['action_mask'].all()
    cfg.dim_model=32;cfg.n_heads=4;cfg.dim_feedforward=64;cfg.n_encoder_layers=1;cfg.n_decoder_layers=1
    learner=MultiViewACT(cfg,initial['stats'],initialize_backbone=False)
    loss,_=learner({k:v.unsqueeze(0) for k,v in sample.items()});loss.backward()
    assert torch.isfinite(loss) and learner.policy.model.action_head.weight.grad.norm()>0
    training.close();validation.close()
    # ACT-only successes remain in TRAIN; 5% VAL still comes entirely from expert episodes.
    model_id=split_episodes(list(range(62)))[1][0]
    model_arrays={k:v.copy() for k,v in arrays.items()}
    model_arrays.update(action_source=np.array(['model']*640),action_segment=np.zeros(640,dtype=np.int64),
        human_active=np.zeros(640,dtype=bool))
    np.savez(root/'model.npz',**model_arrays)
    entries[model_id]['expert_anchor_frames']=[]
    (root/'_collection/manifest.json').write_text(json.dumps(manifest))
    def with_model(source,output,entry,threads,**kwargs):
        item=prepared(source,output,entry,threads,**kwargs)
        if entry['episode_index']==model_id:
            p=root/'model.npz';item['numeric']={'path':os.path.relpath(p,output),'bytes':p.stat().st_size,'sha256':file_digest(p)}
        return item
    def mixed_file(path):
        if int(Path(path).stem)!=model_id:return rollout_file(path)
        a=model_arrays
        weights,frames,validation,report=rollout_windows(a['action_source'],a['action_segment'],a['human_active'],
            a['simulation_time'],a['action'],a[WRENCH],a['action'])
        return a['action'].copy(),weights,frames,validation,report
    monkeypatch.setattr('prepare_expert_only.prepare_episode',with_model)
    monkeypatch.setattr('rollout_sampling.sample_rollout_file',mixed_file)
    cfg.index_path='auto';updated=load_training_index(cfg,initial=initial)
    assert next(e for e in updated['episodes'] if e['episode_index']==model_id)['split']=='train'
    assert all(ep!=model_id for ep,frame in updated['validation_anchors'])
    assert updated['counts']['validation']['episodes']==4 and len(updated['validation_anchors'])==2000
    manifest['attempts']=entries[:2];(root/'_collection/manifest.json').write_text(json.dumps(manifest))
    from prepare_expert_only import prepare_expert_only
    with pytest.raises(ValueError,match='need 2000'):prepare_expert_only(root,initial)


def test_official_model_success_is_accepted_without_fabricating_expert_labels(tmp_path,monkeypatch):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'expert';run=recorded_run(root,expert=True,model_only=True)
    entry=import_human_run(root,run,profile(),tmp_path/'deleted-parent',status=0,expert_source='expert')
    assert entry['accepted'] and entry['outcome']=='model_success'
    assert entry['expert_anchor_frames']==[] and not entry['validation_eligible']
    assert entry['validation']['rollout_training_anchor_count']>0
    assert not (run/'no_expert_windows.parquet').exists()
    sources=pq.read_table(root/entry['parquet'],columns=['action_source']).column(0).to_pylist()
    assert set(sources)=={'model','hold'}
    # Recover an older rejection in place, without increasing attempts or duplicating episodes.
    manifest_path=root/'_collection/manifest.json';manifest=json.loads(manifest_path.read_text())
    rejected=manifest['attempts'][0];rejected.update(accepted=False,rejection_reason='ValueError: No complete human future-20 windows; keep human mode for >1 simulated second')
    (root/entry['parquet']).replace(run/'no_expert_windows.parquet')
    manifest_path.write_text(json.dumps(manifest))
    from human_dataset import recover_successful_runs
    result=recover_successful_runs(root)
    assert len(result)==1 and result[0]['accepted'] and result[0]['attempt']==0
    assert recover_successful_runs(root)==[]
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest['attempts'])==1 and manifest['attempts'][0]['episode_index']==0


def test_recovered_episode_metadata_follows_episode_order_not_attempt_order(tmp_path):
    from collect_sfp0_sideview import refresh_metadata
    attempts=[{'accepted':True,'episode_index':0,'validation':{'frames':30}},
        {'accepted':True,'episode_index':2,'validation':{'frames':40}},
        {'accepted':True,'episode_index':1,'validation':{'frames':50}}]
    refresh_metadata(tmp_path,{'attempts':attempts},profile())
    rows=pq.read_table(tmp_path/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert [r['episode_index'] for r in rows]==[0,1,2]
    assert [r['dataset_from_index'] for r in rows]==[0,30,80]


def test_refresh_teacher_data_transfers_scales_uses_new_only_val2000_and_freezes_snapshot(tmp_path,monkeypatch):
    from prepare_sfp0_act_multiview import prepare
    monkeypatch.setattr('prepare_sfp0_act_multiview.DATA_DRIVE_ROOT',tmp_path)
    base=parent_index(tmp_path/'parent');root=Path(base['dataset_root'])
    entries=[{'accepted':True,'episode_index':i,'attempt':i,'parquet':f'{i}.parquet',
        'validation':{'frames':640,'parquet_sha256':str(i).zfill(64)},
        'official_score':{'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}},
        'recorder':{'teacher_return':True,'writer_error':None}} for i in range(62)]
    manifest=root/'_collection/manifest.json'
    manifest.write_text(json.dumps({'settings':profile(),'task':TASK,'attempts':entries}))
    def cached_episode(source,output,entry,threads):
        item={'episode_index':entry['episode_index'],'source_attempt':entry['attempt'],
              'source_sha256':entry['validation']['parquet_sha256'],'length':640}
        for kind,name in [('images','base.images'),('numeric','base.npz')]:
            p=root/name;item[kind]={'path':os.path.relpath(p,output),'bytes':p.stat().st_size,'sha256':file_digest(p)}
        return item
    monkeypatch.setattr('prepare_sfp0_act_multiview.prepare_episode',cached_episode)
    cfg=MultiViewConfig(dataset_root=str(root),device='cpu',amp='none',refresh_data=True,init_from='parent.pt',eval_samples=2000)
    initial={'config':{'index_path':str(root/'index.json')},'stats':base['stats'],
             'data_index_sha256':base['index_content_sha256']}
    fresh=load_training_index(cfg,initial=initial)
    assert fresh['counts']['train']['episodes']==58 and fresh['counts']['validation']['episodes']==4
    assert fresh['stats']==base['stats'] and len(fresh['validation_anchors'])==2000
    old_split={e['episode_index']:e['split'] for e in base['episodes']}
    assert all(e['split']=='train' for e in fresh['episodes'] if e['episode_index'] in old_split)
    assert all(e>=26 for e,f in fresh['validation_anchors'])
    assert fresh['index_content_sha256']!=base['index_content_sha256']
    # Appended episodes must not enter an already-started run through ordinary resume.
    manifest.write_text(json.dumps({'settings':profile(),'task':TASK,'attempts':entries+[{**entries[-1],'episode_index':62,'attempt':62}]}))
    resumed=load_training_index(cfg,resume={'data_index_sha256':fresh['index_content_sha256']})
    assert len(resumed['episodes'])==62 and resumed['validation_anchors']==fresh['validation_anchors']
    path=Path(cfg.index_path);bad=copy.deepcopy(fresh);bad.pop('_directory')
    train=next(e for e in bad['episodes'] if e['split']=='train' and e['episode_index']<26)
    val=next(e for e in bad['episodes'] if e['split']=='validation')
    train['split'],val['split']='validation','train'
    bad['index_content_sha256']=index_digest(bad);path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='frozen seed'):read_index(path)
    original={k:v for k,v in fresh.items() if k!='_directory'}
    bad=copy.deepcopy(original);bad['stats'][STATE]['mean'][0]+=.01
    bad['index_content_sha256']=index_digest(bad);path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='normalization'):read_index(path)
    path.write_text(json.dumps(original))
    manifest.write_text(json.dumps({'settings':profile(),'task':TASK,'attempts':entries[:26]}))
    with pytest.raises(ValueError,match='No new accepted'):prepare(root,parent=base)
    entries.extend([{**entries[0],'episode_index':i,'attempt':i,
        'validation':{'frames':640,'parquet_sha256':str(i).zfill(64)}} for i in range(62,82)])
    manifest.write_text(json.dumps({'settings':profile(),'task':TASK,'attempts':entries}))
    next_round=prepare(root,parent=fresh,eval_samples=2000)
    assert len(next_round['episodes'])==82 and next_round['counts']['validation']['episodes']==5
    assert next_round['stats']==base['stats']
    previous={e['episode_index']:e['split'] for e in fresh['episodes']}
    assert all(e['split']=='train' for e in next_round['episodes'] if e['episode_index'] in previous)


@pytest.mark.parametrize('rejection',['test','discard','failed_score','incomplete','writer_error'])
def test_nonexpert_or_unsuccessful_runs_are_not_promoted(tmp_path,monkeypatch,rejection):
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    human=tmp_path/'human';run=recorded_run(human)
    if rejection=='failed_score':
        (run/'scoring/scoring.yaml').write_text(yaml.safe_dump({'trial_1':{'tier_1':{'score':1},'tier_3':{'score':0,'message':'No insertion'}}}))
    if rejection=='writer_error':
        p=run/'raw/task_1.json';j=json.loads(p.read_text());j['writer_error']='disk failed';p.write_text(json.dumps(j))
    entry=import_human_run(human,run,profile(),tmp_path/'parent',status=1 if rejection=='incomplete' else 0,
        operator_discarded=rejection=='discard',test_run=rejection=='test')
    assert not entry['accepted'] and entry['rejection_reason']
    assert (run/'raw/task_1.sqlite3').exists() and (run/'raw/task_1.images').exists()
    manifest=json.loads((human/'_collection/manifest.json').read_text())
    assert not any(e.get('accepted') for e in manifest['attempts'])
