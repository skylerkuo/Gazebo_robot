#!/usr/bin/env python3
"""Current SFP0 workflow: collection, scratch ACT, frozen-ACT residual Flow."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parent
MODELS=Path('/mnt/d/robot_models')
DATASETS=Path('/mnt/d/robot_datasets')
ACT_ROOT=MODELS/'aic_sfp0_act_direct_scratch_v1'
FLOW_ROOT=MODELS/'aic_sfp0_flow_recovery_after_direct_act_v1'

COMMANDS={
    'prepare-env':('Prepare the official Gazebo image','prepare_image.py',()),
    'collect':('Collect full official-controller episodes','collect_sfp0_direct.py',()),
    'collect-assisted':('ACT attempt followed by official expert recovery','collect_sfp0_expert.py',
        ('--checkpoint',str(ACT_ROOT/'checkpoints/best.pt'))),
    'prepare-direct':('Curate successful full-controller demonstrations','prepare_sfp0_direct_act.py',()),
    'prepare-recovery':('Curate successful expert-recovery episodes','prepare_sfp0_recovery.py',()),
    'train-act':('Train ACT from scratch on direct demonstrations','train_sfp0_act_direct.py',()),
    'train-flow':('Freeze the new ACT; train residual Flow on recovery','train_sfp0_flow_recovery.py',()),
    'run-act':('Autonomous ACT inference in Gazebo','run_sfp0_multiview_gazebo.py',
        ('--checkpoint',str(ACT_ROOT/'checkpoints/best.pt'))),
    'run-flow':('Autonomous ACT + residual Flow inference','run_sfp0_multiview_gazebo.py',
        ('--checkpoint',str(FLOW_ROOT/'checkpoints/best.pt'),
         '--config',str(ROOT/'configs/sfp0_flow_multiview_gazebo.json'))),
    'evaluate-act':('Repeated autonomous ACT trials with official scoring','evaluate_sfp0_act.py',
        ('--checkpoint',str(ACT_ROOT/'checkpoints/best.pt'),
         '--output-dir',str(ACT_ROOT/'evaluation_100_v1'))),
    'evaluate-flow':('Repeated autonomous ACT + Flow trials','evaluate_sfp0_residual_flow.py',
        ('--checkpoint',str(FLOW_ROOT/'checkpoints/best.pt'),
         '--output-dir',str(FLOW_ROOT/'evaluation_100_v1'))),
}


def build_command(command,extra):
    _,script,defaults=COMMANDS[command]
    return [sys.executable,'-u',str(ROOT/'scripts'/script),*defaults,*extra]


def status():
    """Read metadata/logs only; no tensor loads, simulator calls or cache scans."""
    result={'datasets':[],'training':[]}
    for name in ('aic_sfp0_expert_v1','aic_sfp0_direct_success_v1','aic_sfp0_recovery_only_v1'):
        path=DATASETS/name/'_collection/manifest.json'
        if not path.exists():continue
        value=json.loads(path.read_text());entries=[e for e in value['attempts'] if e.get('accepted')]
        result['datasets'].append({'name':name,'accepted_episodes':len(entries),
            'direct_episodes':sum(e.get('recorder',{}).get('controller')=='official_cheatcode_only' for e in entries)})
    for name in (ACT_ROOT.name,FLOW_ROOT.name):
        path=MODELS/name/'metrics.jsonl'
        if not path.exists():continue
        latest=validation=None
        with path.open() as stream:
            for line in stream:
                try:entry=json.loads(line)
                except ValueError:continue
                if entry.get('type') in ('start','resume','train'):latest=entry
                if entry.get('type')=='validation':validation=entry
        result['training'].append({'name':name,'latest_logged_step':latest.get('step') if latest else None,
            'validation_step':validation.get('step') if validation else None,
            'position_l2_mm':validation.get('position_l2_mm') if validation else None,
            'note':'Log snapshot only; this does not establish that the process is still running.'})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Commands:\n'+'\n'.join(f'  {name:18} {value[0]}' for name,value in COMMANDS.items())+
        '\n  status             Read current dataset counts and latest training metrics\n'
        '\nUse COMMAND --help for its arguments. Put --print-command before COMMAND to preview.')
    parser.add_argument('--print-command',action='store_true',help='Print the command without running it')
    parser.add_argument('command',choices=[*COMMANDS,'status'],nargs='?')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.command is None:parser.print_help();return 0
    if args.command=='status':
        if args.arguments:parser.error('status takes no arguments')
        print(json.dumps(status(),indent=2,ensure_ascii=False));return 0
    command=build_command(args.command,args.arguments)
    if args.print_command:print(shlex.join(command));return 0
    # Replace the wrapper so Ctrl+C, process identity and checkpoint handlers work
    # exactly as they do when the original script is started directly.
    os.execv(sys.executable,command)
    return 0


if __name__=='__main__':raise SystemExit(main())
