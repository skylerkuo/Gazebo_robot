#!/usr/bin/env python3
"""Retain learner motion and expert handoff in existing data; never rewrite raw rows."""
import argparse
import fcntl
import json
from pathlib import Path

import ftact  # Make the existing Arrow reader dependencies available.
from human_dataset import EXPERT_FORMAT,import_receipt_path
from rollout_sampling import ROLLOUT_FILTER,sample_rollout_file
from sideview_data import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1'))
    args=parser.parse_args();root=args.dataset_root.resolve()
    if not root.is_relative_to(Path('/mnt/d')):parser.error('Keep recordings on D:')
    with (root/'_collection/import.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'_collection/manifest.json';manifest=json.loads(path.read_text())
        if manifest['format']!=EXPERT_FORMAT:raise ValueError('Select the model+official-expert dataset')
        reports=[]
        for entry in manifest['attempts']:
            if not entry.get('accepted'):continue
            _,_,train,val,report=sample_rollout_file(root/entry['parquet'])
            entry.update(rollout_train_anchor_frames=train,expert_validation_frames=val,
                rollout_filter=dict(ROLLOUT_FILTER),rollout_sampling_report=report,model_motion_preserved=True)
            entry['validation']['rollout_training_anchor_count']=len(train)
            json_write(import_receipt_path(entry['run'],'expert',entry.get('trial','trial_1')),entry)
            reports.append({'episode':entry['episode_index'],**report})
        manifest['rollout_filter']=dict(ROLLOUT_FILTER)
        manifest['training_rule']='Treat moving model and expert recovery targets equally; boolean mask excludes stationary interiors and hold. VAL uses expert-only windows.'
        json_write(path,manifest)
        json_write(root/'_collection/rollout_sampling_audit.json',{'episodes':reports,
            'settings':ROLLOUT_FILTER,'raw_rows_deleted':False,'timestamps_spliced':False})
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
