#!/usr/bin/env python3
"""Append full official CheatCode demonstrations to the existing assisted dataset.

No ACT checkpoint is loaded. --episodes is the TOTAL accepted episode target,
including existing episodes; rerunning the same command resumes that target.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path

from collect_sfp0_sideview import ROOT, run_batch
from expert_batch import pending_expert_batches, prune_scoring_bags
from human_dataset import EXPERT_FORMAT, import_human_run
from sideview_data import validate_settings
from storage_checks import GIB, ensure_free_space


def collection_state(root):
    value=json.loads((root/'_collection/manifest.json').read_text())
    if value['format']!=EXPERT_FORMAT:
        raise ValueError('Select the existing official-expert intervention dataset')
    return value


def pending_direct_runs(root):
    manifest=collection_state(root)
    known={(e['run'],e.get('trial','trial_1')) for e in manifest['attempts']}
    pending=[]
    for path in sorted((root/'runs').glob('*/settings.json')):
        settings=json.loads(path.read_text())
        if settings.get('recording_mode')!='official_cheatcode_only':continue
        if any((str(path.parent.resolve()),trial) not in known for trial in settings['trial_ids']):
            pending.append(path.parent)
    return pending


def import_direct_run(root,run):
    manifest=collection_state(root)
    settings=json.loads((run/'settings.json').read_text())
    if settings.get('recording_mode')!='official_cheatcode_only':
        raise ValueError('Not a controller-only recording session')
    if any(settings.get(k)!=v for k,v in manifest['settings'].items() if k!='real_time_factor'):
        raise ValueError('Recording settings differ from the existing dataset')
    # Require a completed launcher receipt; do not promote interrupted recordings.
    status=json.loads((run/'launcher.json').read_text())['exit_code']
    entries=[]
    for trial in settings['trial_ids']:
        entry=import_human_run(root,run,manifest['settings'],manifest['parent_collection_root'],
            status=status,expert_source='expert',trial_id=trial,task_id=trial)
        entries.append(entry)
        print(json.dumps({'type':'direct_expert_episode_import','trial':trial,
            'accepted':entry['accepted'],'outcome':entry.get('outcome'),
            'episode_index':entry.get('episode_index'),'reason':entry.get('rejection_reason')}),flush=True)
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1'))
    parser.add_argument('--episodes',type=int,required=True,help='TOTAL accepted episodes including existing data')
    parser.add_argument('--max-attempts',type=int,default=400,help='TOTAL attempts including existing data')
    parser.add_argument('--batch-size',type=int,default=5)
    parser.add_argument('--real-time-factor',type=float,default=.5)
    parser.add_argument('--max-wall-seconds',type=float,default=1800,help='Wall-time budget per shared simulator session')
    parser.add_argument('--gui',choices=('true','false'),default='false')
    parser.add_argument('--keep-scoring-bags',action='store_true')
    parser.add_argument('--import-run',type=Path,help='Recover a finished controller-only batch, then exit')
    args=parser.parse_args()
    if args.episodes<1 or args.max_attempts<args.episodes or args.batch_size<1:
        parser.error('Use positive episode/batch targets and max-attempts >= episodes')
    if not 0<args.real_time_factor<=1 or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=10:
        parser.error('Use 0 < real-time-factor <= 1 and finite max-wall-seconds > 10')
    root=args.output_dir.expanduser().resolve()
    if not root.is_relative_to(Path('/mnt/d')):parser.error('Dataset must be on D:')
    if not (root/'_collection/manifest.json').is_file():parser.error('Select an existing expert dataset')
    if not (ROOT/'runtime/rootfs.ready').exists():parser.error('Prepare the official runtime first')
    # One Gazebo session at a time, including the original assisted collectors.
    with (ROOT/'runtime/session.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another robot_insert Gazebo session is running')
        manifest=collection_state(root)
        validate_settings(manifest['settings'])
        if args.import_run:
            run=args.import_run.resolve()
            if not run.is_relative_to(root/'runs'):parser.error('import-run must be inside this dataset runs folder')
            import_direct_run(root,run)
            if not args.keep_scoring_bags:prune_scoring_bags(run)
            return 0
        pending=pending_expert_batches(root)
        if pending:parser.error(f'Recover assisted batch first with collect_sfp0_expert.py --import-run {pending[0]}')
        pending=pending_direct_runs(root)
        if pending:parser.error(f'Recover controller batch first with --import-run {pending[0]}')
        while True:
            manifest=collection_state(root)
            accepted=sum(bool(e.get('accepted')) for e in manifest['attempts'])
            attempted=len(manifest['attempts'])
            print(json.dumps({'type':'direct_expert_collection_progress','accepted':accepted,
                'target':args.episodes,'remaining':max(0,args.episodes-accepted),
                'attempts':attempted,'controller':'official CheatCode only','act_enabled':False}),flush=True)
            if accepted>=args.episodes:return 0
            count=min(args.batch_size,args.episodes-accepted,args.max_attempts-attempted)
            if count<=0:raise SystemExit('Attempt limit reached; inspect rejected episodes')
            ensure_free_space(root,(count+2)*GIB,'collecting controller demonstrations')
            first=max((e['attempt'] for e in manifest['attempts']),default=-1)+1
            attempts=list(range(first,first+count))
            run,status,interrupted=run_batch(root,attempts,manifest['settings'],args.gui,
                args.max_wall_seconds,args.real_time_factor,official_expert=True)
            entries=import_direct_run(root,run)
            if status==0 and not args.keep_scoring_bags:prune_scoring_bags(run)
            if interrupted:return 130
            if status or not any(e['accepted'] for e in entries):
                raise SystemExit(f'Session failed or produced no usable episodes; inspect {run}')


if __name__=='__main__':raise SystemExit(main())
