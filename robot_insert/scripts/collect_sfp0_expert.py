#!/usr/bin/env python3
"""Collect successful ACT-to-official-expert recovery episodes without keyboard input."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import math
import os
import signal

from automatic_expert_control import EXPERT_MAX_SECONDS,EXPERT_CONTROL_VERSION


def run_collection_session(command):
    """First Ctrl+C completes/imports the active batch; a second aborts it."""
    process=subprocess.Popen(command,start_new_session=True)
    try:return process.wait(),False
    except KeyboardInterrupt:
        print('Stop requested: finishing and importing the current batch, then stopping. '
              'Press Ctrl+C again to abort this batch immediately (raw recordings remain).',flush=True)
        try:return process.wait(),True
        except KeyboardInterrupt:
            if process.poll() is None:os.killpg(process.pid,signal.SIGINT)
            try:return process.wait(timeout=30),True
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                return process.wait(),True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=10,help='Total successful episodes (ACT alone or expert-assisted), including earlier runs')
    parser.add_argument('--repair-only',action='store_true',help='Re-import old official successes rejected for missing expert windows; do not start Gazebo')
    parser.add_argument('--max-attempts',type=int,default=30,help='Total attempted episodes, including earlier runs')
    parser.add_argument('--batch-size',type=int,default=5,help='Independent trials per shared Gazebo/model session')
    parser.add_argument('--real-time-factor',type=float,default=.5,help='Simulation pacing, 0 < factor <= 1; keeps 20 Hz recording')
    parser.add_argument('--import-run',type=Path,help='Recover a completed batch import without starting another simulator')
    parser.add_argument('--keep-scoring-bags',action='store_true',help='Keep large scored ROS bags for later scoring/debugging')
    parser.add_argument('--output-dir',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1'))
    parser.add_argument('--checkpoint',type=Path,default=Path('/mnt/d/robot_models/aic_sfp0_act_multiview_v1/checkpoints/best.pt'))
    parser.add_argument('--gui',choices=('true','false'),default='true')
    parser.add_argument('--expert-after-seconds',type=float,default=20)
    parser.add_argument('--expert-max-seconds',type=float,default=EXPERT_MAX_SECONDS)
    parser.add_argument('--max-wall-seconds',type=float,default=420,help='Wall-time budget per trial; multiplied by actual session batch size')
    args=parser.parse_args()
    if args.episodes<1 or args.max_attempts<args.episodes:parser.error('Use positive episode target and max-attempts >= episodes')
    if args.batch_size<1 or not 0<args.real_time_factor<=1:parser.error('Use batch-size >= 1 and 0 < real-time-factor <= 1')
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=10:parser.error('max-wall-seconds must be finite and > 10')
    if any(not math.isfinite(v) or v<=0 for v in (args.expert_after_seconds,args.expert_max_seconds)):
        parser.error('Expert handoff and time budget must be finite and positive')
    root=args.output_dir.resolve()
    if not root.is_relative_to(Path('/mnt/d')):parser.error('Recovery data must be on D:')
    from expert_batch import import_expert_session,pending_expert_batches,prune_scoring_bags
    if args.import_run:
        run=args.import_run.resolve()
        if not run.is_relative_to(root/'runs'):parser.error('import-run must be inside this collection\'s runs folder')
        import_expert_session(root,run)
        if not args.keep_scoring_bags:prune_scoring_bags(run)
        return 0
    pending=pending_expert_batches(root)
    if pending:parser.error(f'Unimported batch exists: {pending[0]}; recover it with --import-run before collecting more')
    from human_dataset import recover_successful_runs
    for entry in recover_successful_runs(root):
        print(json.dumps({'type':'recovered_episode','run':entry['run'],'accepted':entry['accepted'],
            'outcome':entry.get('outcome'),'reason':entry.get('rejection_reason')}),flush=True)
    if args.repair_only:return 0
    print(json.dumps({'type':'expert_collection_config','controller_version':EXPERT_CONTROL_VERSION,
        'expert_after_seconds':args.expert_after_seconds,'expert_max_seconds':args.expert_max_seconds,
        'batch_size':args.batch_size,'real_time_factor':args.real_time_factor}),flush=True)
    def progress():
        path=root/'_collection/manifest.json'
        if not path.exists():return 0,0
        value=json.loads(path.read_text())
        if value['format']!='aic-sfp0-official-expert-interventions-v1':raise ValueError('Output contains a different dataset')
        return sum(bool(e.get('accepted')) for e in value['attempts']),len(value['attempts'])
    while True:
        accepted,attempts=progress()
        print(json.dumps({'type':'expert_collection_progress','accepted':accepted,'target':args.episodes,'attempts':attempts}),flush=True)
        if accepted>=args.episodes:return 0
        if attempts>=args.max_attempts:raise SystemExit('Attempt limit reached; inspect rejected recovery runs')
        count=min(args.batch_size,args.episodes-accepted,args.max_attempts-attempts)
        from storage_checks import ensure_free_space,GIB
        try:ensure_free_space(root,(count+2)*GIB,'collecting the next expert batch')
        except OSError as error:raise SystemExit(str(error)) from None
        command=[sys.executable,'-u',str(Path(__file__).with_name('run_sfp0_multiview_gazebo.py')),
            '--auto-expert','--expert-data-root',str(root),'--checkpoint',str(args.checkpoint.resolve()),
            '--gui',args.gui,'--expert-after-seconds',str(args.expert_after_seconds),
            '--expert-max-seconds',str(args.expert_max_seconds),
            '--batch-size',str(count),'--real-time-factor',str(args.real_time_factor),
            '--max-wall-seconds',str(args.max_wall_seconds*count)]
        if args.keep_scoring_bags:command.append('--keep-scoring-bags')
        status,stopping=run_collection_session(command)
        if status or progress()[1]<=attempts:
            if stopping:return 130
            raise SystemExit('Recovery session failed; inspect logs before retrying')
        if stopping:return 130


if __name__=='__main__':raise SystemExit(main())
