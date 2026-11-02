#!/usr/bin/env python3
"""Curate successful expert-assisted recovery episodes, preserving real sources."""
import argparse
import json
from pathlib import Path
from prepare_sfp0_direct_act import curate,SOURCE

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=SOURCE)
    parser.add_argument('--output-dir',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_recovery_only_v1'))
    args=parser.parse_args()
    report=curate(args.dataset_root,args.output_dir,recovery=True)
    print(json.dumps({'type':'recovery_dataset_ready',**report},indent=2),flush=True)
