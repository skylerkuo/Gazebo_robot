#!/usr/bin/env python3
"""Freeze available accepted SFP0 episodes: 5% whole-episode VAL, fixed anchors.

Build bounded-memory random-access RGB packs on D:, avoiding a second full-size
Hugging Face Arrow image cache. Source recordings remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from ftact import STATE, WRENCH, json_write
from act_multiview import (DEFAULT_PREPARED, FOUR_CAMERAS, INDEX_FORMAT, VALIDATION_SAMPLES, SUPPORTED_VALIDATION_SAMPLES, file_digest,extended_split,
    index_digest, read_index, split_episodes, training_statistics, validation_anchors)
from sideview_data import successful_score
from storage_checks import ensure_free_space,storage_write_guard,GIB
import pyarrow.parquet as pq
DATA_DRIVE_ROOT=Path('/mnt/d')


def encode_row(row):
    result = []
    for key in FOUR_CAMERAS:
        image = cv2.imdecode(np.frombuffer(row[key]['bytes'],dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError(f'Invalid source RGB: {key}')
        image = cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
        ok,png = cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,1])
        if not ok: raise ValueError('Failed to encode resized image')
        result.append(png.tobytes())
    return result


def prepare_episode(root, output, entry, threads, *, command_provenance=False):
    ep = entry['episode_index']
    source = root/entry['parquet']
    directory = output/'cache'
    directory.mkdir(exist_ok=True)
    metadata = directory/f'episode-{ep:03d}.json'
    if metadata.exists():
        item = json.loads(metadata.read_text())
        if bool(item.get('command_provenance',False))!=command_provenance:
            raise ValueError('Prepared command-provenance format changed; use another cache root')
        if item['source_sha256']!=entry['validation']['parquet_sha256']:
            raise ValueError('Source file changed since cache preparation')
        for kind in ('images','numeric'):
            p = output/item[kind]['path']
            if not p.exists() or p.stat().st_size!=item[kind]['bytes'] or file_digest(p)!=item[kind]['sha256']:
                raise ValueError(f'Changed prepared artifact {p}; use a fresh preparation directory')
        return item
    if file_digest(source)!=entry['validation']['parquet_sha256']:
        raise ValueError(f'Source Parquet checksum mismatch: {source}')
    columns=[STATE,WRENCH,'action','timestamp','simulation_time',
        'action_time','camera_times','joint_time','wrench_time','episode_index','frame_index']
    if command_provenance:columns+=['action_source','action_segment','human_active']
    numeric_table = pq.read_table(source,columns=columns)
    n = len(numeric_table)
    if n!=entry['validation']['frames'] or n<=20:
        raise ValueError('Source episode length differs from collection record')
    raw = numeric_table.to_pydict()
    times = np.asarray(raw['simulation_time'])
    if (np.any(np.diff(times)<=0) or (not command_provenance and not np.allclose(np.diff(times),.05,rtol=0,atol=1e-5))):
        raise ValueError('Source episode is not a complete 20 Hz sequence')
    if raw['episode_index']!=[ep]*n or raw['frame_index']!=list(range(n)):
        raise ValueError('Noncontiguous source episode/frame indices')
    if np.any(np.asarray(raw['action_time'])>times+1e-8):
        raise ValueError('Source action uses a future command')
    ages = times[:,None]-np.asarray(raw['camera_times'])
    if ages.min() < -1e-8 or ages.max()>.012+1e-8:
        raise ValueError('Noncausal or unsynchronized source image')
    for key in ('joint_time','wrench_time'):
        age = times-np.asarray(raw[key])
        if age.min() < -1e-8 or age.max()>.1+1e-8:
            raise ValueError('Noncausal/stale source state or force')
    numeric = {key:np.asarray(raw[key],dtype=np.float32) for key in (STATE,WRENCH,'action')}
    if any(not np.isfinite(v).all() for v in numeric.values()):
        raise ValueError('Nonfinite source state/force/action')
    if not np.allclose(np.linalg.norm(numeric['action'][:,3:],axis=-1),1,atol=1e-4):
        raise ValueError('Source action quaternion is not normalized')
    if command_provenance:
        numeric.update(action_source=np.asarray(raw['action_source'],dtype='U16'),
            action_segment=np.asarray(raw['action_segment'],dtype=np.int64),
            human_active=np.asarray(raw['human_active'],dtype=bool),simulation_time=times)
    # Reserve enough for an uncompressed episode plus headroom. Do not discover
    # a full DrvFS drive only when buffered PNG output is flushed on close.
    ensure_free_space(directory,n*(288*256*3*4+256)+GIB,'writing this episode RGB cache')
    images = directory/f'episode-{ep:03d}.images'
    offsets = np.empty((n,4,2),dtype=np.int64)
    frame = 0
    with storage_write_guard(directory), images.with_suffix('.images.partial').open('wb',buffering=4*1024*1024) as stream, \
            ThreadPoolExecutor(max_workers=threads) as pool:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32,columns=list(FOUR_CAMERAS)):
            for encoded in pool.map(encode_row,batch.to_pylist()):
                for camera, png in enumerate(encoded):
                    offsets[frame,camera] = [stream.tell(),len(png)]
                    stream.write(png)
                frame += 1
    if frame!=n: raise ValueError('RGB frame count differs from numeric frame count')
    images.with_suffix('.images.partial').replace(images)
    arrays = directory/f'episode-{ep:03d}.npz'
    with storage_write_guard(directory), arrays.with_suffix('.npz.partial').open('wb') as stream:
        np.savez(stream,**numeric,image_offsets=offsets)
    arrays.with_suffix('.npz.partial').replace(arrays)
    item = {'episode_index':ep,'source_attempt':entry['attempt'],'length':n,'command_provenance':command_provenance,
            'source_parquet':str(source),'source_sha256':entry['validation']['parquet_sha256']}
    for kind,path in (('images',images),('numeric',arrays)):
        item[kind] = {'path':str(path.relative_to(output)),'bytes':path.stat().st_size,'sha256':file_digest(path)}
    json_write(metadata,item)
    return item


def select_source_episodes(manifest, episodes=None):
    accepted = sorted([e for e in manifest['attempts'] if e.get('accepted')],key=lambda e:e['episode_index'])
    if episodes is not None and (episodes<2 or len(accepted)<episodes):
        raise ValueError(f'Need {episodes} accepted episodes; only {len(accepted)} are available.')
    selected = accepted if episodes is None else accepted[:episodes]
    if len(selected)<2:
        raise ValueError(f'Need at least two accepted episodes; only {len(selected)} are available.')
    ids = [e['episode_index'] for e in selected]
    if ids!=list(range(len(selected))) or len({e['attempt'] for e in selected})!=len(selected):
        raise ValueError('Repeated/missing source episodes or source attempts')
    if not all(successful_score(e['official_score']) and e['recorder']['teacher_return']
               and not e['recorder']['writer_error'] for e in selected):
        raise ValueError('Selected episode lacks successful official scoring/recording')
    return selected


def snapshot_settings(root, selected, seed, eval_samples=VALIDATION_SAMPLES):
    return {'format':INDEX_FORMAT,'dataset_root':str(Path(root).resolve()),'episodes':len(selected),
        'seed':seed,'validation_fraction':.05,'validation_samples':eval_samples,
        'input_shape':[3,256,288],
        'sources':[[e['episode_index'],e['attempt'],e['validation']['parquet_sha256']] for e in selected]}


def prepare(root, output=None, *, episodes=None, seed=42, threads=2, parent=None, eval_samples=VALIDATION_SAMPLES):
    root = Path(root).resolve()
    output = (Path(output) if output is not None else root/'_training'/Path(DEFAULT_PREPARED).name).resolve()
    if not output.is_relative_to(DATA_DRIVE_ROOT):
        raise ValueError('Prepared data/cache must be on D:')
    if threads<1: raise ValueError('Need at least one preparation thread')
    if eval_samples not in SUPPORTED_VALIDATION_SAMPLES:raise ValueError('Use 1000 or 2000 fixed validation anchors')
    # Read one atomic manifest snapshot. Appended episodes belong to a future run.
    manifest = json.loads((root/'_collection/manifest.json').read_text())
    selected = select_source_episodes(manifest,episodes)
    settings = snapshot_settings(root,selected,seed,eval_samples)
    transfer=None
    if parent is not None:
        if root!=Path(parent['dataset_root']).resolve():raise ValueError('New data must extend the parent collection')
        if manifest['task']!=parent['task']:raise ValueError('Task changed since the parent training snapshot')
        if len(selected)<=len(parent['episodes']):raise ValueError('No new accepted episodes yet; omit --refresh-data to use the parent snapshot')
        train_ids,val_ids=extended_split(settings['sources'],parent,seed)
        transfer={'parent_index_path':str(Path(parent['_directory'])/'index.json'),
                  'parent_index_sha256':parent['index_content_sha256'],
                  'split_policy':'validation_from_new_episodes_only'}
        settings['transfer']=transfer
    else:train_ids,val_ids = split_episodes([e['episode_index'] for e in selected],seed)
    available = sum(e['validation']['frames']-20 for e in selected if e['episode_index'] in val_ids)
    if available<eval_samples:
        raise ValueError(f'The {len(val_ids)} held-out episodes (5%, rounded up) have only {available} '
                         f'complete anchors; need {eval_samples}. Collect more episodes first.')
    key = index_digest(settings)[:16]
    snapshot = output/'snapshots'/f'episodes-{len(selected):06d}-{key}'
    print(json.dumps({'type':'data_snapshot','accepted_episodes':len(selected),
        'train_episodes':len(train_ids),'validation_episodes':len(val_ids),
        'validation_samples':eval_samples,'index':str(snapshot/'index.json')}),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'preparation.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another preparation process is using this cache directory') from None
    snapshot.mkdir(parents=True,exist_ok=True)
    settings_path = snapshot/'preparation.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Preparation settings/source membership changed; choose another output directory')
    if (snapshot/'index.json').exists():
        result = read_index(snapshot/'index.json')
        if result['preparation_settings']!=settings: raise ValueError('Existing index uses different source membership')
        return result
    json_write(settings_path,settings)
    if shutil.disk_usage(output).free < 2*1024**3:
        raise ValueError('Less than 2 GiB free on preparation drive')
    cv2.setNumThreads(1)
    items,training = [],[]
    for number,entry in enumerate(selected,1):
        item = prepare_episode(root,output,entry,threads)
        item['split'] = 'train' if item['episode_index'] in train_ids else 'validation'
        # Reuse immutable per-episode RGB packs across different frozen snapshots.
        for kind in ('images','numeric'):
            item[kind]['path'] = os.path.relpath(output/item[kind]['path'],snapshot)
        items.append(item)
        if item['split']=='train' and parent is None:
            with np.load(snapshot/item['numeric']['path'],allow_pickle=False) as arrays:
                training.append({key:arrays[key].copy() for key in (STATE,WRENCH,'action')})
        print(json.dumps({'type':'prepare_episode','prepared':number,'total':len(selected),
                          'episode':item['episode_index'],'split':item['split']}),flush=True)
    value = {'format':INDEX_FORMAT,'dataset_root':str(root),'task':manifest['task'],
        'preparation_settings':settings,'fps':20,'chunk_size':20,'action_offset':1,
        'cameras':list(FOUR_CAMERAS),'input_shape':[3,256,288],
        'image_preprocessing':'OpenCV INTER_AREA to 288x256; lossless PNG RGB cache',
        'validation_fraction':.05,'validation_samples':eval_samples,
        'episodes':items,'test_episodes':[],
        'validation_anchors':validation_anchors([e for e in items if e['split']=='validation'],count=eval_samples),
        'stats':parent['stats'] if parent is not None else training_statistics(training),
        'counts':{name:{'episodes':sum(e['split']==name for e in items),
            'frames':sum(e['length'] for e in items if e['split']==name),
            'anchors':sum(e['length']-20 for e in items if e['split']==name)} for name in ('train','validation')}}
    if transfer is not None:
        value.update(transfer=transfer,normalization_source='frozen parent training statistics for transferred weights')
    value['index_content_sha256'] = index_digest(value)
    json_write(snapshot/'index.json',value)
    return read_index(snapshot/'index.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_multiview_v1'))
    parser.add_argument('--output-dir',type=Path,help='Cache root; indexes go into immutable snapshots/')
    parser.add_argument('--episodes',type=int,help='Optional limit; default uses all currently accepted episodes')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--eval-samples',type=int,choices=SUPPORTED_VALIDATION_SAMPLES,default=2000)
    args = parser.parse_args()
    if (args.episodes is not None and args.episodes<2) or args.threads<1: parser.error('Need positive threads and at least two episodes')
    try:
        index = prepare(args.dataset_root,args.output_dir,episodes=args.episodes,seed=args.seed,threads=args.threads,eval_samples=args.eval_samples)
    except ValueError as error:
        parser.exit(1,f'Dataset preparation stopped: {error}\n')
    print(json.dumps({'type':'prepared','index':str(Path(index['_directory'])/'index.json'),
        'counts':index['counts'],'validation_samples':index['validation_samples']}),flush=True)


if __name__=='__main__': main()
