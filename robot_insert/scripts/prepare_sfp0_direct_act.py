#!/usr/bin/env python3
"""Curate only successful full-CheatCode episodes on D: and freeze an ACT seed."""
import argparse
import copy
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil

from act_multiview import FOUR_CAMERAS,FORMAT,file_digest,index_digest
from collect_sfp0_sideview import refresh_metadata
from human_dataset import EXPERT_FORMAT
from sideview_data import json_write,successful_score

SOURCE=Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1')
DESTINATION=Path('/mnt/d/robot_datasets/aic_sfp0_direct_success_v1')
SEED_SOURCE=Path('/mnt/d/robot_models/aic_sfp0_act_flow_finetune_v2/checkpoints/best_act.pt')
SEED_TARGET=Path('/mnt/d/robot_models/_initializations/aic_sfp0_act_direct_v1/act.pt')
DATA_DRIVE=Path('/mnt/d')


def select_direct(manifest):
    if manifest['format']!=EXPERT_FORMAT:raise ValueError('Expected the official-expert dataset')
    selected=[]
    for entry in manifest['attempts']:
        recorder=entry.get('recorder',{})
        if not entry.get('accepted') or recorder.get('controller')!='official_cheatcode_only':continue
        if (not successful_score(entry['official_score']) or not recorder.get('policy_return')
                or recorder.get('writer_error') or entry.get('test_run') or entry.get('operator_discarded')):
            raise ValueError('Direct episode lacks valid official success/recording')
        if entry['validation'].get('control_source_counts')!={'expert':entry['validation']['frames']}:
            raise ValueError('Direct episode contains non-expert action labels')
        selected.append(entry)
    selected.sort(key=lambda e:e['episode_index'])
    if len(selected)<2:raise ValueError('Need at least two successful direct episodes')
    if len({e['episode_index'] for e in selected})!=len(selected):raise ValueError('Repeated direct episode ID')
    return selected


def select_recovery(manifest):
    if manifest['format']!=EXPERT_FORMAT:raise ValueError('Expected the official-expert dataset')
    selected=[]
    for entry in manifest['attempts']:
        recorder=entry.get('recorder',{})
        if (not entry.get('accepted') or recorder.get('controller')!='model_and_official_expert'
                or not entry.get('expert_validation_frames')):continue
        if (not successful_score(entry['official_score']) or not recorder.get('policy_return')
                or recorder.get('writer_error') or entry.get('test_run') or entry.get('operator_discarded')):
            raise ValueError('Recovery episode lacks valid official success/recording')
        selected.append(entry)
    selected.sort(key=lambda e:e['episode_index'])
    if len(selected)<2:raise ValueError('Need at least two successful expert-recovery episodes')
    return selected


def materialize(source,target,*,expected_bytes=None):
    """Independent directory entries; hard-linked immutable packs avoid duplication."""
    source=Path(source);target=Path(target)
    if source.is_symlink() or not source.is_file():raise ValueError(f'Expected regular source file: {source}')
    size=source.stat().st_size
    if expected_bytes is not None and size!=expected_bytes:raise ValueError(f'Source size changed: {source}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.stat().st_size!=size:raise ValueError(f'Existing target differs: {target}')
        if not os.path.samefile(source,target) and file_digest(source)!=file_digest(target):
            raise ValueError(f'Existing target differs: {target}')
        return
    try:os.link(source,target)
    except OSError as error:
        if error.errno not in (errno.EXDEV,errno.EPERM,errno.EOPNOTSUPP):raise
        temporary=target.with_suffix(target.suffix+'.partial')
        shutil.copyfile(source,temporary)
        if file_digest(source)!=file_digest(temporary):raise ValueError('Copy verification failed')
        temporary.replace(target)


def curate(source=SOURCE,destination=DESTINATION,*,recovery=False):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if not source.is_relative_to(DATA_DRIVE) or not destination.is_relative_to(DATA_DRIVE):
        raise ValueError('Both datasets must be on D:')
    if source==destination:raise ValueError('Use a separate destination folder')
    manifest=json.loads((source/'_collection/manifest.json').read_text())
    entries=select_recovery(manifest) if recovery else select_direct(manifest)
    selection=('accepted model_and_official_expert; successful official score; complete expert recovery windows'
        if recovery else 'accepted official_cheatcode_only; successful official score; exclusively expert commands')
    plan={'source_dataset':str(source),'selection':selection,
        'source_episodes':[[e['episode_index'],e['attempt'],e['validation']['parquet_sha256']] for e in entries],
        'episode_ids_preserved':True,'training_only_curated_dataset':True}
    plan_path=destination/'_collection/curation.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Source membership changed; choose a fresh destination')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination is nonempty without a matching curation plan')
    else:json_write(plan_path,plan)
    output=copy.deepcopy(manifest);output['attempts']=[];output['curation']=plan
    cache_source=source/'_training/act_corrections/cache'
    cache_target=destination/'_training/act_corrections/cache'
    for number,entry in enumerate(entries,1):
        item=copy.deepcopy(entry)
        source_parquet=source/entry['parquet'];target_parquet=destination/entry['parquet']
        materialize(source_parquet,target_parquet)
        ep=entry['episode_index'];metadata=cache_source/f'episode-{ep:03d}.json'
        if not metadata.exists():
            from prepare_sfp0_act_multiview import prepare_episode
            prepared=source/'_training/act_corrections'
            prepared.mkdir(parents=True,exist_ok=True)
            with (prepared/'preparation.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not metadata.exists():
                    print(json.dumps({'type':'prepare_new_curated_episode','episode':ep}),flush=True)
                    prepare_episode(source,prepared,entry,2,command_provenance=True)
        cached=json.loads(metadata.read_text())
        if cached['source_sha256']!=entry['validation']['parquet_sha256'] or not cached.get('command_provenance'):
            raise ValueError('Prepared episode differs from successful direct source')
        for kind in ('numeric','images'):
            original=source/'_training/act_corrections'/cached[kind]['path']
            target=destination/'_training/act_corrections'/cached[kind]['path']
            materialize(original,target,expected_bytes=cached[kind]['bytes'])
        cached['source_parquet']=str(target_parquet)
        json_write(cache_target/metadata.name,cached)
        run=Path(entry['run']);new_run=destination/'runs'/run.name
        task=entry.get('recording_task_id','task_1')
        # Keep the calibration/scoring proof portable without duplicating ROS bags.
        files=['settings.json','provenance.json','run_manifest.json','launcher.json','scoring/scoring.yaml',
            f'raw/{task}.json',f'raw/{task}_calibration.json']
        for relative in files:
            old=run/relative
            if not old.exists():
                if relative in (f'raw/{task}_calibration.json','scoring/scoring.yaml'):
                    raise ValueError(f'Missing direct episode calibration/proof: {old}')
                continue
            new=new_run/relative;new.parent.mkdir(parents=True,exist_ok=True)
            if not new.exists():shutil.copyfile(old,new)
        item.update(run=str(new_run),source_run=str(run),source_dataset_root=str(source))
        output['attempts'].append(item)
        print(json.dumps({'type':'curate_recovery_episode' if recovery else 'curate_direct_episode','prepared':number,'total':len(entries),
            'source_episode':ep}),flush=True)
    refresh_metadata(destination,output,output['settings'])
    info=json.loads((destination/'meta/info.json').read_text())
    original_info=json.loads((source/'meta/info.json').read_text())
    for feature in ('action_source','action_segment','human_active'):
        info['features'][feature]=original_info['features'][feature]
    info.update(source_episode_ids=[e['episode_index'] for e in entries],curated_direct_only=not recovery,
        curated_recovery_only=recovery)
    json_write(destination/'meta/info.json',info)
    return {'dataset_root':str(destination),'episodes':len(entries),'frames':info['total_frames'],
        'episode_ids':[e['episode_index'] for e in entries],'data_files_use_hardlinks_when_supported':True}


def freeze_act_seed(source=SEED_SOURCE,target=SEED_TARGET):
    import torch
    target=Path(target);receipt=target.with_suffix('.origin.json')
    if not target.exists():materialize(source,target)
    payload=torch.load(target,map_location='cpu',weights_only=True)
    if payload.get('format')!=FORMAT or payload.get('camera_order')!=list(FOUR_CAMERAS) or payload.get('smoke_test'):
        raise ValueError('Select a real four-view ACT checkpoint')
    value={'source':str(Path(source).resolve()),'frozen_checkpoint':str(target.resolve()),
        'step':payload['step'],'sha256':file_digest(target),'data_index_sha256':payload['data_index_sha256']}
    if receipt.exists() and json.loads(receipt.read_text())!=value:raise ValueError('ACT initialization snapshot changed')
    json_write(receipt,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=SOURCE)
    parser.add_argument('--output-dir',type=Path,default=DESTINATION)
    parser.add_argument('--init-from',type=Path,help='Optional legacy fine-tune seed; omitted for from-scratch ACT')
    args=parser.parse_args()
    report=curate(args.dataset_root,args.output_dir)
    seed=None
    if args.init_from:
        seed=freeze_act_seed(args.init_from)
        json_write(Path(report['dataset_root'])/'_collection/act_training_seed.json',seed)
    print(json.dumps({'type':'direct_act_ready',**report,'initialization':seed},indent=2),flush=True)


if __name__=='__main__':main()
