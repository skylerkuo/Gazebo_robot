#!/usr/bin/env python3
"""Read-only integrity check of the collected four-view episodes and official scores."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/dataset_python'))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from sideview_data import CAMERAS, digest_file, successful_score


def validate(root, *, verify_hashes=True):
    root = Path(root)
    manifest = json.loads((root/'_collection/manifest.json').read_text())
    settings = manifest['settings']
    episodes = [a for a in manifest['attempts'] if a.get('accepted')]
    if not episodes:
        raise ValueError('No accepted episodes')
    offset, summaries = 0, []
    for ep, entry in enumerate(episodes):
        path = root/entry['parquet']
        if not successful_score(entry['official_score']):
            raise ValueError('An accepted episode lacks successful official scoring')
        if verify_hashes and digest_file(path) != entry['validation']['parquet_sha256']:
            raise ValueError(f'Parquet checksum changed: {path}')
        schema = pq.read_schema(path)
        features = json.loads(schema.metadata[b'huggingface'])['info']['features']
        if not all(features[f'observation.images.{name}_camera']['_type']=='Image' for name in CAMERAS):
            raise ValueError('Missing Hugging Face image feature metadata')
        table = pq.read_table(path, columns=['simulation_time','action_time','timestamp','camera_times',
            'joint_time','wrench_time','action','observation.tcp_pose','observation.arm_joint_positions',
            'observation.environment_state','episode_index','frame_index','index'])
        r = table.to_pydict()
        n = len(table)
        if n != entry['validation']['frames'] or n < settings['minimum_frames']:
            raise ValueError('Episode frame count differs from manifest')
        if not np.all(np.asarray(r['episode_index'])==ep):
            raise ValueError('Episode index mismatch')
        if r['frame_index'] != list(range(n)) or r['index'] != list(range(offset,offset+n)):
            raise ValueError('Noncontiguous frame/global indices')
        times = np.asarray(r['simulation_time'])
        gaps = np.diff(times)
        if np.any(gaps<=0) or gaps.max()>settings['maximum_frame_gap_s']:
            raise ValueError('Nonmonotonic timestamps or a dropped frame')
        if not np.allclose(r['timestamp'], times-times[0], atol=1e-5):
            raise ValueError('Episode timestamp differs from simulation clock')
        if np.any(np.asarray(r['action_time'])>times+1e-8):
            raise ValueError('Future command used as a current action label')
        ages = times[:,None]-np.asarray(r['camera_times'])
        if ages.min() < -1e-8 or ages.max()>settings['camera_tolerance_s']+1e-8:
            raise ValueError('Noncausal or stale image')
        for key in ('joint_time','wrench_time'):
            age = times-np.asarray(r[key])
            if age.min() < -1e-8 or age.max()>settings['state_tolerance_s']+1e-8:
                raise ValueError('Noncausal or stale joint/wrench')
        for key,width in [('action',7),('observation.tcp_pose',7),('observation.arm_joint_positions',6),
                          ('observation.environment_state',6)]:
            array = np.asarray(r[key])
            if array.shape != (n,width) or not np.isfinite(array).all():
                raise ValueError(f'Invalid numeric column {key}')
            if width==7 and not np.allclose(np.linalg.norm(array[:,3:],axis=-1),1,atol=1e-4):
                raise ValueError('Invalid quaternion norm')
        for name in CAMERAS:
            column = pq.read_table(path, columns=[f'observation.images.{name}_camera']).column(0)
            digest = hashlib.sha256()
            for frame,value in enumerate(column.to_pylist()):
                digest.update(value['bytes'])
                with Image.open(io.BytesIO(value['bytes'])) as image:
                    if image.mode!='RGB' or image.size!=(settings['record_width'],settings['record_height']):
                        raise ValueError('Image dimensions or encoding changed')
                    image.verify()
            if digest.hexdigest()!=entry['validation']['camera_stream_sha256'][name]:
                raise ValueError('Image stream checksum changed')
        summaries.append({'episode':ep,'frames':n,'duration_s':float(times[-1]-times[0]),
                          'maximum_gap_s':float(gaps.max()),'maximum_camera_age_s':float(ages.max())})
        offset += n
    info = json.loads((root/'meta/info.json').read_text())
    if info['total_episodes']!=len(episodes) or info['total_frames']!=offset:
        raise ValueError('Dataset metadata totals do not match the recordings')
    return {'passed':True,'accepted_episodes':len(episodes),'frames':offset,'cameras':list(CAMERAS),
            'all_official_insertions_successful':True,'all_images_verified':True,
            'causal_alignment_verified':True,'episodes':summaries,
            'scope':'Collection integrity, not learned-policy accuracy or insertion success.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path,nargs='?',default=Path('/mnt/d/robot_datasets/aic_sfp0_multiview_v1'))
    args=parser.parse_args()
    print(json.dumps(validate(args.dataset),indent=2))


if __name__=='__main__':
    main()
