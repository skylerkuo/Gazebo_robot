"""Add expert intervention windows while preserving the parent model's VAL and scales."""
import copy
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from act_multiview import CORRECTION_INDEX_FORMAT,index_digest,read_index
from human_control import expert_anchor_frames
from human_dataset import HUMAN_FORMAT,EXPERT_FORMAT
from prepare_sfp0_act_multiview import prepare_episode
from sideview_data import json_write,successful_score
from idle_sampling import DEFAULT_IDLE_FILTER,filter_idle_anchors
DATA_DRIVE_ROOT=Path('/mnt/d')


def prepare_corrections(base,root,*,allow_test=False):
    root=Path(root).resolve()
    if not root.is_relative_to(DATA_DRIVE_ROOT):raise ValueError('Correction data/cache must be on D:')
    manifest_path=root/'_collection/manifest.json'
    if not manifest_path.exists():raise ValueError('No human recordings yet; finish a successful keyboard-assisted episode first')
    manifest=json.loads(manifest_path.read_text())
    if manifest['format'] not in (HUMAN_FORMAT,EXPERT_FORMAT) or Path(manifest['parent_collection_root']).resolve()!=Path(base['dataset_root']).resolve():
        raise ValueError('Human corrections come from a different camera/task collection')
    synthetic=bool(manifest.get('synthetic_fixture',False))
    expert_source='expert' if manifest['format']==EXPERT_FORMAT else 'human'
    entries=sorted([e for e in manifest['attempts'] if e.get('accepted')],key=lambda e:e['episode_index'])
    if not entries:raise ValueError('No successful human correction episodes yet; collect and finish a human-assisted episode first')
    if (synthetic or any(e.get('test_run') for e in entries)) and not allow_test:
        raise ValueError('Test recordings cannot be used for formal training')
    if len({e['episode_index'] for e in entries})!=len(entries):raise ValueError('Repeated human episode index')
    if any((not (synthetic and allow_test) and not successful_score(e['official_score'])) or not e['recorder'].get('policy_return')
           or e['recorder'].get('writer_error') or e.get('operator_discarded') for e in entries):
        raise ValueError('Correction episode lacks successful official scoring/recording')
    reference=json.loads((Path(base['dataset_root'])/'_collection/manifest.json').read_text())['settings']
    if any(manifest['settings'].get(k)!=v for k,v in reference.items() if k not in ('minimum_frames','real_time_factor')):
        raise ValueError('Correction camera/timing settings differ from the parent data')
    settings={'base_index_sha256':base['index_content_sha256'],'root':str(root),
        'expert_source':expert_source,
        'idle_filter':dict(DEFAULT_IDLE_FILTER),
        'sources':[[e['episode_index'],e['validation']['parquet_sha256']] for e in entries],
        'label_rule':f'all future 20 {expert_source}, one control segment, >=1 active correction, 20 Hz continuous',
        'test_only':synthetic or any(e.get('test_run') for e in entries)}
    output=root/'_training/act_corrections';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'preparation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    snapshot=output/'snapshots'/index_digest(settings)[:16];snapshot.mkdir(parents=True,exist_ok=True)
    if (snapshot/'index.json').exists():return read_index(snapshot/'index.json')
    ignored={'_directory','index_content_sha256','correction_episodes','correction_settings','test_only'}
    value={k:copy.deepcopy(v) for k,v in base.items() if k not in ignored}
    for item in value['episodes']:
        for kind in ('numeric','images'):
            item[kind]['path']=os.path.relpath(Path(base['_directory'])/item[kind]['path'],snapshot)
    corrections=[];next_id=max(e['episode_index'] for e in value['episodes'])+1
    for n,entry in enumerate(entries):
        item=prepare_episode(root,output,entry,2,command_provenance=True)
        with np.load(output/item['numeric']['path'],allow_pickle=False) as arrays:
            anchors=expert_anchor_frames(arrays['action_source'],arrays['action_segment'],arrays['human_active'],arrays['simulation_time'],source=expert_source)
            filtered,sampling=filter_idle_anchors(anchors,arrays['action'],arrays['observation.environment_state'],arrays['simulation_time'])
        if not anchors:raise ValueError('Accepted correction has no usable human future-20 windows')
        if anchors!=entry['expert_anchor_frames']:raise ValueError('Stored expert windows differ from the source provenance')
        anchors=filtered
        item.update(source_correction_episode_index=item['episode_index'],expert_source=expert_source,episode_index=next_id+n,
                    split='train',train_anchor_frames=anchors,source_group=entry['run'],
                    idle_filter=dict(DEFAULT_IDLE_FILTER),sampling_report=sampling)
        for kind in ('numeric','images'):item[kind]['path']=os.path.relpath(output/item[kind]['path'],snapshot)
        corrections.append(item)
        print(json.dumps({'type':f'prepare_{expert_source}_correction','prepared':n+1,'total':len(entries),
                          'source_episode':entry['episode_index'],'expert_anchors':len(anchors)}),flush=True)
    value.update(format=CORRECTION_INDEX_FORMAT,correction_episodes=corrections,
        correction_settings=settings,test_only=settings['test_only'],
        normalization_source='frozen parent training statistics; no rescaling of transferred weights')
    train=[e for e in value['episodes'] if e['split']=='train']
    value['counts']['train']={'episodes':len(train)+len(corrections),
        'frames':sum(e['length'] for e in [*train,*corrections]),
        'anchors':sum(e['length']-20 for e in train)+sum(len(e['train_anchor_frames']) for e in corrections),
        f'{expert_source}_episodes':len(corrections),f'{expert_source}_anchors':sum(len(e['train_anchor_frames']) for e in corrections)}
    value['index_content_sha256']=index_digest(value)
    json_write(snapshot/'index.json',value)
    return read_index(snapshot/'index.json')
