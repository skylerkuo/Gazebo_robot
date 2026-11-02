#!/usr/bin/env python3
"""Resumable autonomous evaluation of an alternating ACT + residual Flow pair."""
from evaluate_sfp0_pace import main

if __name__ == '__main__':
    raise SystemExit(main(policy_type='alternating_residual_flow',
        default_model_root='/mnt/d/robot_models/aic_sfp0_alternating_residual_flow_scratch_v1'))
