#!/usr/bin/env python3
"""Freeze the new direct-demo ACT BEST and train residual Flow on expert recovery."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_flow_recovery_after_direct_act.json',
         expected_policy='residual_flow')
