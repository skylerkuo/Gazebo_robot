"""Train only on the retained expert dataset, without the deleted teacher dataset."""
import copy
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from act_multiview import (EXPERT_ONLY_INDEX_FORMAT,ROLLOUT_INDEX_FORMAT,FOUR_CAMERAS,index_digest,read_index,file_digest,
    split_episodes,validation_anchors,training_statistics)
from ftact import STATE,WRENCH
from human_control import expert_anchor_frames
from human_dataset import EXPERT_FORMAT
from idle_sampling import DEFAULT_IDLE_FILTER,filter_idle_anchors
from prepare_sfp0_act_multiview import prepare_episode
from sideview_data import TASK,json_write,successful_score

DATA_DRIVE_ROOT=Path('/mnt/d')


def inspect_expert_collection(root,*,eval_samples=2000,seed=42,retain_model_motion=False):
    root=Path(root).resolve()
    if not root.is_relative_to(DATA_DRIVE_ROOT):raise ValueError('Expert dataset must be on D:')
    manifest=json.loads((root/'_collection/manifest.json').read_text())
    if manifest['format']!=EXPERT_FORMAT or manifest['task']!=TASK:raise ValueError('Expected the official-expert SFP0 dataset')
    entries=sorted([e for e in manifest['attempts'] if e.get('accepted')],key=lambda e:e['episode_index'])
    if manifest.get('synthetic_fixture') or any(e.get('test_run') for e in entries):raise ValueError('Test data is not allowed for expert-only training')
    if any(not successful_score(e['official_score']) or e['recorder'].get('writer_error') or
           not e['recorder'].get('policy_return') or e.get('operator_discarded') for e in entries):
        raise ValueError('Expert episode lacks successful official scoring/recording')
    ids=[e['episode_index'] for e in entries]
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Need at least two different successful expert episodes')
    rollout_data={}
    if retain_model_motion:
        from rollout_sampling import ROLLOUT_FILTER,sample_rollout_file
        for e in entries:rollout_data[e['episode_index']]=sample_rollout_file(root/e['parquet'])
        candidates=[i for i in ids if rollout_data[i][3]]
        train,val=split_episodes(ids,seed,validation_candidates=candidates)
        available=sum(len(rollout_data[e['episode_index']][3]) for e in entries if e['episode_index'] in val)
    else:
        entries=[e for e in entries if e['expert_anchor_frames']]
        ids=[e['episode_index'] for e in entries]
        train,val=split_episodes(ids,seed)
        available=sum(len(e.get('filtered_expert_anchor_frames',e['expert_anchor_frames'])) for e in entries if e['episode_index'] in val)
    report={'accepted_episodes':len(entries),'train_episodes':len(train),'validation_episodes':len(val),
        'validation_episode_ids':val,'eligible_validation_anchors':available,'required_validation_anchors':eval_samples,
        'ready':available>=eval_samples,'validation_fraction':.05,'seed':seed}
    if retain_model_motion:
        report.update(train_anchors=sum(len(rollout_data[i][2]) for i in train),
            retained_model_targets=sum(rollout_data[i][4]['retained_model_targets'] for i in train),
            expert_targets=sum(rollout_data[i][4]['expert_targets'] for i in train),
            masked_stationary_model_targets=sum(rollout_data[i][4]['masked_stationary_model_targets'] for i in train))
    return {'root':root,'entries':entries,'ids':ids,'train':train,'val':val,'rollout_data':rollout_data,
        'candidates':candidates if retain_model_motion else None,'report':report}


def prepare_expert_only(root,initial=None,*,eval_samples=2000,seed=42,retain_model_motion=False,progress=False):
    from rollout_sampling import ROLLOUT_FILTER
    data=inspect_expert_collection(root,eval_samples=eval_samples,seed=seed,retain_model_motion=retain_model_motion)
    root,entries,ids,train,val=(data[k] for k in ('root','entries','ids','train','val'))
    rollout_data,candidates=data['rollout_data'],data['candidates']
    available=data['report']['eligible_validation_anchors']
    print(json.dumps({'type':'assisted_data_readiness',**data['report']}),flush=True)
    if not data['report']['ready']:
        raise ValueError(f'Expert-only data: {len(entries)} episodes, {len(val)} VAL episodes contain only {available} '
                         f'eligible action chunks; need {eval_samples}. Collect more successful expert episodes first.')
    settings={'episodes':len(entries),'seed':seed,'validation_samples':eval_samples,'validation_fraction':.05,
        'sources':[[e['episode_index'],e['attempt'],e['validation']['parquet_sha256']] for e in entries],
        'kind':'expert_only','idle_filter':dict(DEFAULT_IDLE_FILTER)}
    if initial is None:
        settings['normalization_source']='current training episodes only'
    else:settings['normalization_parent_index_sha256']=initial['data_index_sha256']
    if retain_model_motion:settings.update(kind='assisted_rollout',rollout_filter=dict(ROLLOUT_FILTER))
    if retain_model_motion and len(candidates)!=len(ids):settings['validation_candidate_episodes']=candidates
    output=root/'_training/act_corrections';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'preparation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    folder=output/'snapshots'/(('assisted-rollout-' if retain_model_motion else 'expert-only-')+index_digest(settings)[:16]);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'index.json').exists():
        index=read_index(folder/'index.json',progress=progress)
        if initial is not None and index['stats']!=initial['stats']:raise ValueError('Parent normalization statistics changed')
        return index
    items=[];training=[]
    for prepared_count,entry in enumerate(entries,1):
        item=prepare_episode(root,output,entry,2,command_provenance=True)
        with np.load(output/item['numeric']['path'],allow_pickle=False) as a:
            raw=expert_anchor_frames(a['action_source'],a['action_segment'],a['human_active'],a['simulation_time'],source='expert')
            filtered,report=filter_idle_anchors(raw,a['action'],a['observation.environment_state'],a['simulation_time'])
            if initial is None and item['episode_index'] in train:
                training.append({key:a[key].copy() for key in (STATE,WRENCH,'action')})
        if raw!=entry['expert_anchor_frames'] or (not retain_model_motion and not filtered):raise ValueError('Expert-window provenance changed or is empty')
        item.update(split='train' if item['episode_index'] in train else 'validation',expert_source='expert',
            train_anchor_frames=filtered,idle_filter=dict(DEFAULT_IDLE_FILTER),sampling_report=report)
        if retain_model_motion:
            tcp,weights,full_train,expert_val,full_report=rollout_data[entry['episode_index']]
            motion_key=index_digest({'source_sha256':item['source_sha256'],'filter':ROLLOUT_FILTER})[:16]
            path=output/'cache'/f"episode-{entry['episode_index']:03d}-motion-{motion_key}.npz"
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                temporary=path.with_suffix('.npz.partial')
                with temporary.open('wb') as stream:np.savez(stream,tcp_pose=tcp,action_mask=weights.astype(bool))
                temporary.replace(path)
            item.update(train_anchor_frames=full_train,expert_validation_frames=expert_val,
                rollout_filter=dict(ROLLOUT_FILTER),sampling_report=full_report,
                motion={'path':os.path.relpath(path,folder),'bytes':path.stat().st_size,'sha256':file_digest(path)})
        for kind in ('images','numeric'):item[kind]['path']=os.path.relpath(output/item[kind]['path'],folder)
        items.append(item)
        print(json.dumps({'type':'prepare_assisted_episode','prepared':prepared_count,'total':len(entries),
            'episode':item['episode_index'],'split':item['split'],'frames':item['length']}),flush=True)
    val_items=[dict(e,train_anchor_frames=e['expert_validation_frames']) if retain_model_motion else e for e in items if e['split']=='validation']
    value={'format':ROLLOUT_INDEX_FORMAT if retain_model_motion else EXPERT_ONLY_INDEX_FORMAT,'dataset_root':str(root),'task':TASK,'fps':20,'chunk_size':20,
        'action_offset':1,'cameras':list(FOUR_CAMERAS),'input_shape':[3,256,288],
        'validation_samples':eval_samples,'validation_fraction':.05,'preparation_settings':settings,
        'episodes':items,'test_episodes':[],
        'validation_anchors':validation_anchors(val_items,count=eval_samples),
        'stats':training_statistics(training) if initial is None else copy.deepcopy(initial['stats']),
        'counts':{name:{'episodes':sum(e['split']==name for e in items),
            'frames':sum(e['length'] for e in items if e['split']==name),
            'anchors':sum(len(e['train_anchor_frames']) for e in items if e['split']==name)} for name in ('train','validation')}}
    if initial is None:
        value.update(normalization_source='current training episodes only',normalization_training_episode_ids=sorted(train))
    else:
        value.update(normalization_parent_index_sha256=initial['data_index_sha256'],
            normalization_source='frozen initialized checkpoint statistics; old training data not required')
    value['index_content_sha256']=index_digest(value);json_write(folder/'index.json',value)
    return read_index(folder/'index.json',progress=progress)
