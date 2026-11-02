#!/usr/bin/env python3
"""Autonomous frozen ACT + learned residual; no ground-truth expert takeover."""
from pathlib import Path
from run_sfp0_multiview_gazebo import main,ROOT

if __name__=='__main__':
    raise SystemExit(main(
        default_checkpoint=Path('/mnt/d/robot_models/aic_sfp0_residual_flow_expert_v1/checkpoints/best.pt'),
        default_config=ROOT/'configs/sfp0_flow_multiview_gazebo.json'))
