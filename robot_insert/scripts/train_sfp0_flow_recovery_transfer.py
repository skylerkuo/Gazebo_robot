#!/usr/bin/env python3
"""Freeze this experiment's new ACT and transfer its encoder to recovery Flow."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__ == '__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_flow_recovery_direct100_transfer.json',
         expected_policy='residual_flow')
